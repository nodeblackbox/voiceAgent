"""Offline Parakeet benchmark: load time, per-utterance latency, RTFx, VRAM, WER vs Handy reference.

Usage:
    python bench/bench_offline.py                 # CUDA fp32 (default)
    python bench/bench_offline.py --device cpu --quant int8
    python bench/bench_offline.py --repeats 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (AUDIO_DIR, MODEL_NAME, assert_cuda_still_active, gpu_mem_mib, load_parakeet, load_wav_16k, pct,
                    proc_gpu_mem_mib, save_json, wer)

con = Console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "hybrid", "cpu"], default="hybrid")
    ap.add_argument("--quant", default=None, help="e.g. int8 (CPU only; CUDA int8 is slow)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timestamps", action="store_true", help="also request word timestamps")
    args = ap.parse_args()

    ref = json.loads((AUDIO_DIR / "handy_reference.json").read_text())
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        sys.exit("no wavs in audio/")

    before = gpu_mem_mib()
    con.print(f"[bold]Loading {MODEL_NAME}[/] device={args.device} quant={args.quant}")
    model, load_s, active = load_parakeet(args.device, args.quant)
    after = gpu_mem_mib()
    con.print(f"loaded in {load_s:.2f}s  active providers={active}")
    if before and after:
        con.print(f"GPU mem (whole card): {before[0]} -> {after[0]} MiB of {after[1]}")
    if args.timestamps:
        model = model.with_timestamps()

    # Warm-up (first call compiles CUDA kernels / allocates workspace)
    warm = load_wav_16k(wavs[0])
    t0 = time.perf_counter()
    model.recognize(warm[: 16000 * 2])
    con.print(f"warm-up call: {time.perf_counter() - t0:.3f}s")
    active = assert_cuda_still_active(model, args.device)
    con.print(f"providers after warm-up: {active}")

    rows = []
    for w in wavs:
        audio = load_wav_16k(w)
        dur = len(audio) / 16000
        lat = []
        text = ""
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = model.recognize(audio)
            lat.append(time.perf_counter() - t0)
            text = out.text if hasattr(out, "text") else out
        ref_text = ref.get(w.name, "")
        row = {
            "file": w.name,
            "audio_s": round(dur, 2),
            "lat_min_s": round(min(lat), 4),
            "lat_med_s": round(float(np.median(lat)), 4),
            "rtfx": round(dur / min(lat), 1),
            "wer_vs_handy": round(wer(ref_text, text), 3) if ref_text else None,
            "hyp": text,
            "ref": ref_text,
        }
        if args.timestamps and hasattr(out, "timestamps") and out.timestamps:
            row["first_word_ts"] = out.timestamps[0]
            row["n_tokens"] = len(out.timestamps)
        rows.append(row)

    proc_mem = proc_gpu_mem_mib()
    total = gpu_mem_mib()

    t = Table(title=f"{MODEL_NAME} on {args.device} ({args.quant or 'fp32'}), best of {args.repeats}")
    for c in ["file", "audio_s", "lat_min_s", "lat_med_s", "rtfx", "wer_vs_handy"]:
        t.add_column(c, justify="right")
    for r in rows:
        t.add_row(r["file"], str(r["audio_s"]), str(r["lat_min_s"]), str(r["lat_med_s"]), str(r["rtfx"]),
                  "-" if r["wer_vs_handy"] is None else str(r["wer_vs_handy"]))
    con.print(t)

    for r in rows:
        con.print(f"\n[bold cyan]{r['file']}[/]")
        con.print(f"  HYP: {r['hyp']}")
        if r["ref"]:
            con.print(f"  REF: {r['ref']}")

    summary = {
        "model": MODEL_NAME,
        "device": args.device,
        "quant": args.quant,
        "active_providers": active,
        "load_s": round(load_s, 3),
        "process_gpu_mib": proc_mem,
        "gpu_used_total_mib": total,
        "lat_p50_s": round(pct([r["lat_med_s"] for r in rows], 50), 4),
        "lat_p95_s": round(pct([r["lat_med_s"] for r in rows], 95), 4),
        "rows": rows,
    }
    out = save_json(f"offline_{args.device}_{args.quant or 'fp32'}.json", summary)
    con.print(f"\nprocess GPU mem: {proc_mem} MiB   whole-card: {total}")
    con.print(f"saved -> {out}")


if __name__ == "__main__":
    main()
