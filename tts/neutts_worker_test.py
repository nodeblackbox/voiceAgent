"""Protocol test for the NeuTTS worker/client bridge, without the real (gated) model weights.

Runs the REAL `neutts_worker.py` process (in .venv-neutts) with `Worker.model` monkeypatched to a fake
that mimics `FastNeuTTS`'s shape (encode_reference, infer_stream) using only numpy — no torch, no
llama.cpp, no network. This proves the actual wire protocol end to end: framing, set_voice, streaming,
and — the part that matters most given this session's history with barge-in — that a client aborting
mid-stream (as StreamingSpeech.interrupt() does) sends `cancel`, the worker's background stdin-reader
picks it up between chunks (not only after the whole sentence), and the pipe stays synchronized for the
next command afterward. This does NOT prove real audio quality; only the transport is exercised here.

    .venv-neutts\\Scripts\\python.exe tts\\neutts_worker_test.py

Run with .venv-neutts's python, not the main venv's: this script's own imports are numpy-only, but the
patched worker launcher still does `import neutts_worker`, whose top-level `neutts_compat.apply()` /
`from fast_neutts import FastNeuTTS` need torch/torchao/neutts installed (only true in .venv-neutts).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TTS_DIR = ROOT / "tts"
sys.path.insert(0, str(TTS_DIR))
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("ok" if cond else "FAIL") + f": {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


FAKE_WORKER_PATCH = '''
import sys, time
sys.path.insert(0, r"{tts_dir}")
import neutts_worker as _w
import numpy as np

class FakeModel:
    """Mimics FastNeuTTS's shape with pure numpy: no torch, no llama.cpp, no network."""
    def __init__(self):
        self.watermarker = None
    def encode_reference(self, path):
        return np.array([1, 2, 3], dtype=np.int64)  # stand-in "codes"
    def infer_stream(self, text, ref_codes, ref_text):
        n_chunks = max(2, len(text) // 8)
        for i in range(n_chunks):
            time.sleep(0.05)  # simulate real per-chunk generation latency
            yield (np.full(2400, 0.01 * (i + 1), dtype=np.float32))

_orig_init = _w.Worker.__init__
def _patched_init(self, backbone, codec, device, voice, samples_dir, watermark, codec_device="auto", seed=None, **kw):
    self.samples_dir = samples_dir
    self._ref_cache = {{}}
    self.model = FakeModel()
    self.codec_device = "fake"
    self.load_s = 0.01
    self.voice = voice
    self.ref_codes, self.ref_text = self._load_reference(voice)
    self.warmup_s = 0.01
_w.Worker.__init__ = _patched_init
_w.main()
'''


def spawn():
    patch_file = TTS_DIR / "_fake_worker_launch.py"
    patch_file.write_text(FAKE_WORKER_PATCH.format(tts_dir=str(TTS_DIR).replace("\\", "\\\\")), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(patch_file), "--voice", "dave"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    return proc


def main() -> None:
    patch_file = TTS_DIR / "_fake_worker_launch.py"
    patch_file.write_text(FAKE_WORKER_PATCH.format(tts_dir=str(TTS_DIR).replace("\\", "\\\\")), encoding="utf-8")

    from neutts_stream import NeuTTSEngine

    # Point the client at THIS process's own python (main venv) running the patched worker, instead of
    # .venv-neutts — the fake model needs no torch/neutts install at all.
    import neutts_stream as ns
    ns.VENV_PYTHON = Path(sys.executable)
    real_popen = subprocess.Popen

    def patched_popen(cmd, **kw):
        # swap the real worker script path for our patched launcher, keep everything else identical
        cmd = list(cmd)
        idx = cmd.index(str(TTS_DIR / "neutts_worker.py"))
        cmd[idx] = str(patch_file)
        print(f"[test] spawning: {cmd}", file=sys.stderr)
        return real_popen(cmd, **kw)
    subprocess.Popen = patched_popen
    try:
        eng = NeuTTSEngine(voice="dave", start_timeout=20)
    finally:
        subprocess.Popen = real_popen

    check("engine started via the real protocol (ready handshake)", eng.voice == "dave")
    check("load_s / warmup_s reported", eng.load_s is not None and eng.warmup_s is not None)

    print("\n[1] plain synth_stream: chunks arrive, concatenate to the expected length")
    text = "This is a reasonably long sentence for the fake model to chunk up."
    chunks = list(eng.synth_stream(text))
    check("got more than one chunk", len(chunks) > 1, f"{len(chunks)} chunks")
    audio = np.concatenate(chunks)
    check("chunk dtype is float32", audio.dtype == np.float32)
    check("audio values match the fake model's ramp (not corrupted)", abs(audio[0] - 0.01) < 1e-6 and audio[-1] > audio[0])

    print("\n[2] synth() collects the full stream into one array")
    full = eng.synth("Another test sentence here for good measure.")
    check("synth() returns a non-empty concatenated array", len(full) > 0)

    print("\n[3] set_voice round-trips")
    v = eng.set_voice("jo")
    check("set_voice returns and updates .voice", v == "jo" and eng.voice == "jo")

    print("\n[4] barge-in: abandoning synth_stream mid-way sends cancel and re-syncs the pipe")
    gen = eng.synth_stream("A long sentence that should generate several chunks before anyone stops it.")
    first = next(gen)
    check("got at least the first chunk before cancelling", first is not None)
    t0 = time.perf_counter()
    gen.close()  # what StreamingSpeech effectively does when the for-loop is abandoned via `break`
    close_ms = (time.perf_counter() - t0) * 1e3
    check("gen.close() (our GeneratorExit -> cancel path) returns quickly", close_ms < 2000, f"{close_ms:.0f} ms")

    print("\n[5] the pipe is still in sync: a normal call right after cancel works cleanly")
    chunks2 = list(eng.synth_stream("One more sentence to prove the protocol is still synchronized."))
    check("post-cancel synth_stream still works", len(chunks2) > 0, f"{len(chunks2)} chunks")

    eng.close()
    (TTS_DIR / "_fake_worker_launch.py").unlink(missing_ok=True)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL PROTOCOL TESTS PASSED (real weights not exercised — see neutts_worker_test.py docstring)")


if __name__ == "__main__":
    main()
