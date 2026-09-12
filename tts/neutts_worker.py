"""NeuTTS worker process: runs inside `.venv-neutts` (a separate venv from the main agent — neutts
needs torch>=2.8, which is incompatible with the torch==2.6.0+cu124 that Kokoro/Parakeet/the agent's
main venv depend on). Talks to the parent over stdin/stdout with a tiny length-prefixed protocol so the
main-venv `NeuTTSEngine` client (see neutts_stream.py) can drive it exactly like an in-process engine.

Protocol (all on stdout, mixed with nothing else — logging goes to stderr):
    one JSON line per command result/event, optionally followed by raw bytes when a "chunk" event
    carries audio:
        {"event": "ready", "voice": "...", "load_s": .., "warmup_s": .., "vram_mib": ..}
        {"event": "chunk", "n": <float32 sample count>}   <-- immediately followed by n*4 raw bytes
        {"event": "done"}                                  <-- ends one synth_stream call
        {"event": "voice_set", "voice": "..."}
        {"event": "error", "message": "..."}

Commands on stdin, one JSON line each:
        {"cmd": "synth_stream", "text": "...", "id": 1}
        {"cmd": "set_voice", "spec": "dave", "id": 2}
        {"cmd": "cancel"}          # stop whatever is generating right now, ASAP (barge-in)
        {"cmd": "quit"}

`cancel` is read on a background thread (a plain `for line in sys.stdin` loop can't be interrupted from
outside while blocked mid-generation), so it takes effect within about one internal audio chunk
(~0.5 s) of NeuTTS's own streaming, not after the whole sentence finishes — the same barge-in
responsiveness this project already relies on for Kokoro.

Run standalone for a smoke test:
    .venv-neutts\\Scripts\\python.exe tts\\neutts_worker.py --voice dave --backbone neuphonic/neutts-air-q8-gguf
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

TTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TTS_DIR))

# HF_HOME (-> D:\hf-cache, where the gated Neuphonic weights already live) and HF_TOKEN are Windows User
# env vars; this subprocess inherits whatever its parent had, which may be a stale shell that never saw
# them. Read them from the registry before anything touches huggingface_hub. See winenv.py.
from winenv import load_user_env  # noqa: E402

load_user_env(["HF_HOME", "HF_TOKEN", "HF_HUB_OFFLINE"])

# The protocol owns stdout. neutts itself print()s to stdout ("Loading backbone from: ...", and
# "Using seed N" on EVERY synth), and llama.cpp/tqdm write to the C-level fds — any of that landing
# on our pipe desynchronises the JSON+binary framing (the first symptom was the client reporting
# "no response from worker": it had read "Loading backbone from..." as the ready line). So: keep a
# private copy of the real stdout fd for emit()/emit_chunk(), then point fd 1 (and sys.stdout) at
# stderr so every stray print, from any library, goes to the log instead.
_PROTO_FD = os.dup(sys.stdout.fileno())
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
_proto = os.fdopen(_PROTO_FD, "wb", buffering=0)

import neutts_compat  # noqa: E402

neutts_compat.apply()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fast_neutts import FastNeuTTS  # noqa: E402


def resolve_reference(spec: str, samples_dir: Path) -> tuple[Path, str]:
    if "|" in spec:
        wav_str, text_part = spec.split("|", 1)
        wav_path = Path(wav_str)
        if not wav_path.exists():
            raise ValueError(f"reference audio not found: {wav_path}")
        text_path = Path(text_part)
        ref_text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else text_part.strip()
        return wav_path, ref_text
    p = Path(spec)
    if p.suffix.lower() == ".wav" and p.exists():
        txt_path = p.with_suffix(".txt")
        if not txt_path.exists():
            raise ValueError(f"no transcript found next to {p} (expected {txt_path})")
        return p, txt_path.read_text(encoding="utf-8").strip()
    wav_path, txt_path = samples_dir / f"{spec}.wav", samples_dir / f"{spec}.txt"
    if not wav_path.exists():
        names = ", ".join(sorted(x.stem for x in samples_dir.glob("*.wav"))) if samples_dir.exists() else "none"
        raise ValueError(f"unknown voice '{spec}': not a wav path, not a bundled sample ({names})")
    return wav_path, txt_path.read_text(encoding="utf-8").strip()


def emit(obj: dict) -> None:
    _proto.write((json.dumps(obj) + "\n").encode("utf-8"))


def emit_chunk(audio: np.ndarray) -> None:
    data = np.asarray(audio, dtype=np.float32).tobytes()
    emit({"event": "chunk", "n": len(audio)})
    _proto.write(data)


class Worker:
    def __init__(self, backbone: str, codec: str, device: str, voice: str, samples_dir: Path, watermark: bool,
                 codec_device: str = "auto", seed: int | None = None):
        self.samples_dir = samples_dir
        self._ref_cache: dict[str, tuple] = {}
        t0 = time.perf_counter()
        # backbone = llama.cpp (its own CUDA build, independent of torch). codec = torch: on a CUDA torch
        # it can share the GPU, on a CPU-only torch (the original .venv-neutts) it must stay on CPU.
        if codec_device == "auto":
            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.codec_device = codec_device
        # seed=None -> neutts draws a fresh seed per call (and prints it, to stderr now); a fixed seed
        # makes repeated runs of the same text comparable for benchmarking.
        self.model = FastNeuTTS(backbone_repo=backbone, backbone_device=device, codec_repo=codec,
                                codec_device=codec_device, seed=seed)
        if not watermark:
            self.model.watermarker = None
        self.load_s = time.perf_counter() - t0
        self.voice = voice
        self.ref_codes, self.ref_text = self._load_reference(voice)
        t0 = time.perf_counter()
        list(self.model.infer_stream("Warm up.", self.ref_codes, self.ref_text))
        self.warmup_s = time.perf_counter() - t0

    def _load_reference(self, spec: str):
        if spec in self._ref_cache:
            return self._ref_cache[spec]
        wav_path, ref_text = resolve_reference(spec, self.samples_dir)
        cache_path = wav_path.with_suffix(".pt")
        codes = None
        if cache_path.exists():
            # .pt files shipped with the NeuTTS repo samples were pickled with older objects that
            # torch>=2.6's weights_only=True default rejects (UnpicklingError). These are local files
            # we put here ourselves, so fall back to a full load; if even that isn't a tensor, drop
            # the cache and re-encode from the wav (a few hundred ms, once).
            try:
                codes = torch.load(cache_path)
            except Exception:  # noqa: BLE001
                try:
                    codes = torch.load(cache_path, weights_only=False)
                except Exception:  # noqa: BLE001
                    codes = None
            if codes is not None and not torch.is_tensor(codes):
                codes = None
        if codes is None:
            codes = self.model.encode_reference(wav_path)
            torch.save(codes, cache_path)
        self._ref_cache[spec] = (codes, ref_text)
        return codes, ref_text

    def vram_mib(self) -> float:
        # torch here has no CUDA (codec runs on CPU); the backbone's VRAM is llama.cpp's own allocation,
        # not visible through torch — report via nvidia-smi in the client instead if needed.
        return 0.0

    def _read_stdin(self, cmd_q: queue.Queue, cancel: threading.Event) -> None:
        """Runs on its own thread so a 'cancel' arriving mid-generation is seen immediately, not only
        after the current command finishes (a plain blocking `for line in sys.stdin` can't do that)."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cmd.get("cmd") == "cancel":
                cancel.set()
            else:
                cmd_q.put(cmd)
        # stdin closed: the parent exited or died. Stop any generation in flight right now (otherwise a
        # mid-sentence worker would keep the GPU busy until that sentence finished) and shut down.
        cancel.set()
        cmd_q.put({"cmd": "quit"})

    def run(self) -> None:
        emit({"event": "ready", "voice": self.voice, "load_s": self.load_s, "warmup_s": self.warmup_s,
              "codec_device": self.codec_device, "torch": torch.__version__, "python": sys.executable})
        cmd_q: queue.Queue = queue.Queue()
        cancel = threading.Event()
        threading.Thread(target=self._read_stdin, args=(cmd_q, cancel), daemon=True).start()
        while True:
            cmd = cmd_q.get()
            cancel.clear()
            try:
                if cmd["cmd"] == "quit":
                    return
                if cmd["cmd"] == "set_voice":
                    self.ref_codes, self.ref_text = self._load_reference(cmd["spec"])
                    self.voice = cmd["spec"]
                    emit({"event": "voice_set", "voice": self.voice, "id": cmd.get("id")})
                elif cmd["cmd"] == "synth_stream":
                    cancelled = False
                    for chunk in self.model.infer_stream(cmd["text"], self.ref_codes, self.ref_text):
                        if cancel.is_set():
                            cancelled = True
                            break
                        if chunk is not None and len(chunk):
                            emit_chunk(chunk)
                    emit({"event": "cancelled" if cancelled else "done", "id": cmd.get("id")})
            except Exception as e:  # noqa: BLE001
                emit({"event": "error", "id": cmd.get("id"), "message": f"{type(e).__name__}: {e}"})

    def close(self) -> None:
        """Free the llama.cpp backbone explicitly. Leaving it to the interpreter-exit destructor is what
        produces the noisy `llama_free_model ... KeyboardInterrupt` traceback on shutdown."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "close"):
            try:
                backbone.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="neuphonic/neutts-air-q8-gguf")
    ap.add_argument("--codec", default="neuphonic/neucodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice", default="dave")
    ap.add_argument("--samples-dir", default=str(TTS_DIR / "neutts_samples"))
    ap.add_argument("--no-watermark", action="store_true", default=True)
    ap.add_argument("--codec-device", default="auto", help="cuda | cpu | auto (cuda if this venv's torch has it)")
    ap.add_argument("--seed", type=int, default=None, help="fixed sampling seed (default: fresh per call)")
    args = ap.parse_args()
    try:
        w = Worker(args.backbone, args.codec, args.device, args.voice, Path(args.samples_dir), not args.no_watermark,
                   codec_device=args.codec_device, seed=args.seed)
    except Exception as e:  # noqa: BLE001
        emit({"event": "error", "message": f"startup failed: {type(e).__name__}: {e}"})
        sys.exit(1)
    try:
        w.run()
    except KeyboardInterrupt:
        pass  # console Ctrl+C in plain mode reaches every process on the console; exit quietly
    finally:
        w.close()


if __name__ == "__main__":
    main()
