"""Smart Turn v3.2 (Pipecat) end-of-turn test on CPU via onnxruntime.

Two experiments:
  1. Each Handy recording as a whole: does the model think the speaker is done? (they are)
  2. The 64 s monologue cut at every Silero pause (300 ms hangover): each cut is a point where a
     dumb silence-only endpointer WOULD have fired. Smart Turn should say "not done" for most of them
     and "done" only at the real end. Per-call cost is also measured.

Usage: python bench/turn_test.py [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from common import AUDIO_DIR, RESULTS_DIR, load_wav_16k, pct, save_json
from whisper_features import compute_whisper_log_mel_features

con = Console()
SR = 16000
WIN = 8 * SR
MODEL = Path(__file__).resolve().parent.parent / "models" / "smart-turn-v3.2-cpu.onnx"


class SmartTurn:
    def __init__(self, path: Path = MODEL):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> float:
        """Probability that the turn is complete. audio: float32 16 kHz mono, any length."""
        a = audio[-WIN:] if len(audio) > WIN else np.pad(audio, (WIN - len(audio), 0))  # pad at front, keep tail
        feats = compute_whisper_log_mel_features(a.astype(np.float32), do_normalize=True)
        out = self.sess.run(None, {"input_features": feats[None].astype(np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    st = SmartTurn()
    st(np.zeros(SR, dtype=np.float32))  # warm-up
    cost = []

    con.print("[bold]1. Whole recordings (speaker really is done at the end)[/]")
    t = Table()
    for c in ["file", "audio_s", "p(complete)", "verdict", "ms"]:
        t.add_column(c, justify="right")
    rows1 = []
    for w in sorted(AUDIO_DIR.glob("*.wav")):
        a = load_wav_16k(w)
        t0 = time.perf_counter()
        p = st(a)
        ms = (time.perf_counter() - t0) * 1e3
        cost.append(ms)
        v = "DONE" if p >= args.threshold else "not done"
        rows1.append({"file": w.name, "audio_s": round(len(a) / SR, 2), "p": round(p, 3), "verdict": v, "ms": round(ms, 1)})
        t.add_row(w.name, f"{len(a)/SR:.2f}", f"{p:.3f}", v, f"{ms:.1f}")
    con.print(t)

    con.print("\n[bold]2. Monologue cut at each Silero pause (where silence-only endpointing fires)[/]")
    vad = json.loads((RESULTS_DIR / "vad_files.json").read_text())
    long = next(r for r in vad["rows"] if r["file"] == "handy-1785143065.wav")
    a = load_wav_16k(AUDIO_DIR / long["file"])
    t = Table()
    for c in ["cut #", "cut at s", "p(complete)", "verdict", "ms"]:
        t.add_column(c, justify="right")
    rows2 = []
    for i, (s, e) in enumerate(long["segs"], 1):
        clip = a[: int(e * SR)]
        t0 = time.perf_counter()
        p = st(clip)
        ms = (time.perf_counter() - t0) * 1e3
        cost.append(ms)
        last = i == len(long["segs"])
        v = "DONE" if p >= args.threshold else "not done"
        rows2.append({"cut": i, "at_s": e, "p": round(p, 3), "verdict": v, "is_real_end": last, "ms": round(ms, 1)})
        t.add_row(str(i), f"{e:.2f}" + (" (real end)" if last else ""), f"{p:.3f}", v, f"{ms:.1f}")
    con.print(t)
    held = sum(1 for r in rows2 if not r["is_real_end"] and r["verdict"] == "not done")
    con.print(f"held the floor at {held}/{len(rows2)-1} mid-monologue pauses; "
              f"real end -> {rows2[-1]['verdict']} (p={rows2[-1]['p']})")
    con.print(f"per-call cost: p50 {pct(cost,50):.1f} ms  p95 {pct(cost,95):.1f} ms (CPU, 8 s window)")
    out = save_json("turn_test.json", {"threshold": args.threshold, "whole": rows1, "cuts": rows2,
                                        "cost_p50_ms": pct(cost, 50), "cost_p95_ms": pct(cost, 95)})
    con.print(f"saved -> {out}")


if __name__ == "__main__":
    main()
