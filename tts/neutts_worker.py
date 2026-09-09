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

# See the matching comment in agent/talk.py: HF_HOME is a persisted Windows User env var (D: has room,
# C: doesn't). This worker is spawned as a subprocess that inherits whatever env its parent had, so set
# it defensively here too, before the backbone/codec downloads that would otherwise land on C:.
if os.name == "nt" and os.path.isdir(r"D:\hf-cache\huggingface"):
    os.environ.setdefault("HF_HOME", r"D:\hf-cache\huggingface")

TTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TTS_DIR))
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
    sys.stdout.buffer.write((json.dumps(obj) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def emit_chunk(audio: np.ndarray) -> None:
    data = np.asarray(audio, dtype=np.float32).tobytes()
    emit({"event": "chunk", "n": len(audio)})
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


class Worker:
    def __init__(self, backbone: str, codec: str, device: str, voice: str, samples_dir: Path, watermark: bool):
        self.samples_dir = samples_dir
        self._ref_cache: dict[str, tuple] = {}
        t0 = time.perf_counter()
        self.model = FastNeuTTS(backbone_repo=backbone, backbone_device=device, codec_repo=codec, codec_device="cpu")
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
        codes = torch.load(cache_path) if cache_path.exists() else self.model.encode_reference(wav_path)
        if not cache_path.exists():
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
        cmd_q.put({"cmd": "quit"})  # stdin closed (parent died/exited): shut down cleanly

    def run(self) -> None:
        emit({"event": "ready", "voice": self.voice, "load_s": self.load_s, "warmup_s": self.warmup_s})
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="neuphonic/neutts-air-q8-gguf")
    ap.add_argument("--codec", default="neuphonic/neucodec")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice", default="dave")
    ap.add_argument("--samples-dir", default=str(TTS_DIR / "neutts_samples"))
    ap.add_argument("--no-watermark", action="store_true", default=True)
    args = ap.parse_args()
    try:
        w = Worker(args.backbone, args.codec, args.device, args.voice, Path(args.samples_dir), not args.no_watermark)
    except Exception as e:  # noqa: BLE001
        emit({"event": "error", "message": f"startup failed: {type(e).__name__}: {e}"})
        sys.exit(1)
    w.run()


if __name__ == "__main__":
    main()
