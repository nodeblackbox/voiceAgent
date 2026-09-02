"""Break one Parakeet call into preprocessor / encoder / decoder-loop time for cuda vs hybrid vs cpu.

Usage: python bench/profile_stages.py [--devices hybrid cuda cpu] [--file audio/handy-1787878656.wav]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AUDIO_DIR, assert_cuda_still_active, load_parakeet, load_wav_16k, save_json

con = Console()


def profile(model, audio: np.ndarray, repeats: int) -> dict:
    asr = model.asr
    best = {"pre": 1e9, "enc": 1e9, "dec": 1e9, "total": 1e9}
    frames = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        feats, lens = asr._preprocessor([audio], np.array([len(audio)], dtype=np.int64))
        t1 = time.perf_counter()
        enc, enc_lens = asr._encode(feats, lens)
        t2 = time.perf_counter()
        list(asr._decoding(enc, enc_lens))
        t3 = time.perf_counter()
        frames = int(enc_lens[0])
        cur = {"pre": t1 - t0, "enc": t2 - t1, "dec": t3 - t2, "total": t3 - t0}
        best = {k: min(best[k], cur[k]) for k in best}
    best["enc_frames"] = frames
    best["dec_ms_per_frame"] = best["dec"] * 1e3 / max(frames, 1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["hybrid", "cuda", "cpu"])
    ap.add_argument("--files", nargs="+", default=["handy-1787878656.wav", "handy-1785143065.wav"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    results = []
    for dev in args.devices:
        quant = "int8" if dev == "cpu" else None
        model, load_s, active = load_parakeet(dev, quant)
        con.print(f"[bold]{dev}[/] ({quant or 'fp32'}) loaded {load_s:.1f}s providers={active}")
        model.recognize(np.zeros(16000, dtype=np.float32))
        assert_cuda_still_active(model, dev)
        for f in args.files:
            audio = load_wav_16k(AUDIO_DIR / f)
            r = profile(model, audio, args.repeats)
            r.update(device=dev, quant=quant or "fp32", file=f, audio_s=round(len(audio) / 16000, 2))
            results.append(r)
        del model

    t = Table(title="Stage timing (best of N), seconds")
    for c in ["device", "file", "audio_s", "pre", "enc", "dec", "total", "frames", "dec ms/frame", "RTFx"]:
        t.add_column(c, justify="right")
    for r in results:
        t.add_row(r["device"], r["file"], str(r["audio_s"]), f"{r['pre']:.3f}", f"{r['enc']:.3f}", f"{r['dec']:.3f}",
                  f"{r['total']:.3f}", str(r["enc_frames"]), f"{r['dec_ms_per_frame']:.2f}",
                  f"{r['audio_s'] / r['total']:.0f}")
    con.print(t)
    con.print(f"saved -> {save_json('profile_stages.json', results)}")


if __name__ == "__main__":
    main()
