"""Silero VAD test: segments + per-chunk cost on the recorded wavs, or a live mic probability meter.

Usage:
    python bench/vad_test.py                       # run over audio/*.wav
    python bench/vad_test.py --live [--device 52]  # live meter from mic (Ctrl+C to stop)
    python bench/vad_test.py --threshold 0.6 --min-silence 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve_input_device, wasapi_settings, AUDIO_DIR, load_wav_16k, pct, save_json

con = Console()
SR = 16000
CHUNK = 512  # 32 ms at 16 kHz; silero v5/v6 require exactly 512 samples


def load_vad(onnx: bool):
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)


def run_files(args) -> None:
    from silero_vad import VADIterator

    model = load_vad(args.onnx)
    rows = []
    for w in sorted(AUDIO_DIR.glob("*.wav")):
        audio = load_wav_16k(w)
        model.reset_states()
        it = VADIterator(model, threshold=args.threshold, sampling_rate=SR,
                         min_silence_duration_ms=args.min_silence, speech_pad_ms=args.pad)
        segs, cur, cost, probs = [], None, [], []
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            chunk = torch.from_numpy(audio[i:i + CHUNK])
            t0 = time.perf_counter()
            ev = it(chunk, return_seconds=True)
            cost.append(time.perf_counter() - t0)
            if ev and "start" in ev:
                cur = ev["start"]
            elif ev and "end" in ev and cur is not None:
                segs.append((cur, ev["end"]))
                cur = None
        if cur is not None:
            segs.append((cur, len(audio) / SR))
        speech = sum(e - s for s, e in segs)
        rows.append({
            "file": w.name, "audio_s": round(len(audio) / SR, 2), "segments": len(segs),
            "speech_s": round(speech, 2), "first_speech_s": round(segs[0][0], 3) if segs else None,
            "last_end_s": round(segs[-1][1], 3) if segs else None,
            "chunk_cost_p50_ms": round(pct(cost, 50) * 1e3, 3), "chunk_cost_p95_ms": round(pct(cost, 95) * 1e3, 3),
            "segs": [(round(s, 2), round(e, 2)) for s, e in segs],
        })

    t = Table(title=f"Silero VAD ({'onnx' if args.onnx else 'jit'}) thr={args.threshold} min_sil={args.min_silence}ms pad={args.pad}ms")
    for c in ["file", "audio_s", "segments", "speech_s", "first_speech_s", "last_end_s", "chunk p50 ms", "chunk p95 ms"]:
        t.add_column(c, justify="right")
    for r in rows:
        t.add_row(r["file"], str(r["audio_s"]), str(r["segments"]), str(r["speech_s"]), str(r["first_speech_s"]),
                  str(r["last_end_s"]), str(r["chunk_cost_p50_ms"]), str(r["chunk_cost_p95_ms"]))
    con.print(t)
    for r in rows:
        con.print(f"  {r['file']}: {r['segs']}")
    out = save_json("vad_files.json", {"threshold": args.threshold, "min_silence_ms": args.min_silence,
                                        "pad_ms": args.pad, "onnx": args.onnx, "rows": rows})
    con.print(f"saved -> {out}")


def run_live(args) -> None:
    import sounddevice as sd

    model = load_vad(args.onnx)
    con.print(f"[bold]Live VAD meter[/] device={args.device} (Ctrl+C to stop). Bar = speech probability.")
    buf = np.zeros(0, dtype=np.float32)
    state = {"speaking": False, "t_on": 0.0, "last_speech": 0.0}

    def cb(indata, frames, t, status):
        nonlocal buf
        if status:
            con.print(f"[red]{status}[/]")
        buf = np.concatenate([buf, indata[:, 0].copy()])

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                        device=args.device, extra_settings=wasapi_settings(args.device), callback=cb):
        t_start = time.perf_counter()
        try:
            while True:
                if len(buf) >= CHUNK:
                    chunk, buf = buf[:CHUNK], buf[CHUNK:]
                    p = model(torch.from_numpy(chunk), SR).item()
                    now = time.perf_counter() - t_start
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    on = p >= args.threshold
                    if on:
                        state["last_speech"] = now
                    if on and not state["speaking"]:
                        state["speaking"], state["t_on"] = True, now
                        con.print(f"\n[green]SPEECH START[/] @ {now:7.2f}s")
                    elif state["speaking"] and (now - state["last_speech"]) * 1e3 >= args.min_silence:
                        state["speaking"] = False
                        con.print(f"\n[yellow]SPEECH END[/]   @ {now:7.2f}s  (len {now - state['t_on']:.2f}s, hangover {args.min_silence}ms)")
                    bar = "#" * int(p * 40)
                    sys.stdout.write(f"\r{now:7.2f}s p={p:4.2f} rms={rms:6.4f} |{bar:<40}|")
                    sys.stdout.flush()
                else:
                    time.sleep(0.005)
                if args.duration and (time.perf_counter() - t_start) > args.duration:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--device", default="Yeti", help="input device: name substring (default Yeti) or index")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-silence", type=int, default=300, help="hangover ms before END fires")
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--onnx", action="store_true", help="use onnxruntime backend instead of torch jit")
    ap.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = until Ctrl+C)")
    args = ap.parse_args()
    args.device = resolve_input_device(args.device)
    run_live(args) if args.live else run_files(args)


if __name__ == "__main__":
    main()
