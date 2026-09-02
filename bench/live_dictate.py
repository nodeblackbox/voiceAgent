"""Live dictation loop: mic -> ring-buffer pre-roll -> Silero VAD gate -> Parakeet one-pass -> terminal.

Measures, per utterance:
  speech_end -> text printed   (the number that matters for a voice agent)
  asr_infer_s                  (pure Parakeet time)
  utterance length, VAD trigger times

Usage:
    python bench/live_dictate.py                 # default input device
    python bench/live_dictate.py --device 52     # Yeti via WASAPI
    python bench/live_dictate.py --min-silence 500 --preroll 1.0 --threshold 0.5
Ctrl+C to stop. Every utterance is appended to results/live_<timestamp>.jsonl and audio saved to results/utt_*.wav.
"""
from __future__ import annotations

import argparse
import collections
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve_input_device, wasapi_settings, MODEL_NAME, RESULTS_DIR, assert_cuda_still_active, load_parakeet, pct, proc_gpu_mem_mib

con = Console()
SR = 16000
CHUNK = 512  # 32 ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Yeti", help="input device: name substring (default Yeti) or index")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-silence", type=int, default=600, help="hangover ms of silence that ends an utterance (your pauses measured 0.3-0.6 s)")
    ap.add_argument("--min-speech", type=int, default=250, help="ignore blips shorter than this (ms)")
    ap.add_argument("--preroll", type=float, default=1.0, help="seconds of audio kept before VAD trigger")
    ap.add_argument("--max-utt", type=float, default=30.0, help="hard cap: force-cut utterance after N s")
    ap.add_argument("--cpu", action="store_true", help="run Parakeet on CPU int8 instead of hybrid (CUDA encoder + CPU decoder)")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--wav", type=str, default=None, help="simulate the mic: replay this wav in real time instead of capturing")
    args = ap.parse_args()
    args.device = resolve_input_device(args.device)

    from silero_vad import load_silero_vad

    vad = load_silero_vad()
    con.print(f"[bold]Loading {MODEL_NAME}[/] ...")
    dev = "cpu" if args.cpu else "hybrid"
    model, load_s, active = load_parakeet(dev, "int8" if args.cpu else None)
    con.print(f"loaded in {load_s:.1f}s providers={active}")
    model.recognize(np.zeros(SR, dtype=np.float32))  # warm-up
    con.print(f"providers after warm-up: {assert_cuda_still_active(model, dev)}")
    con.print(f"process GPU mem after warm-up: {proc_gpu_mem_mib()} MiB")

    log_path = RESULTS_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    asr_q: queue.Queue[tuple] = queue.Queue()
    lat_e2e, lat_asr = [], []

    def mic_cb(indata, frames, t, status):
        if status:
            con.print(f"[red]mic: {status}[/]")
        audio_q.put(indata[:, 0].copy())

    def asr_worker():
        n = 0
        while True:
            item = asr_q.get()
            if item is None:
                return
            n += 1
            audio, t_speech_end, t_start, t_end_wall = item
            t0 = time.perf_counter()
            text = model.recognize(audio)
            t1 = time.perf_counter()
            e2e = t1 - t_speech_end
            lat_e2e.append(e2e)
            lat_asr.append(t1 - t0)
            rec = {"n": n, "utt_s": round(len(audio) / SR, 2), "asr_infer_s": round(t1 - t0, 3),
                   "speech_end_to_text_s": round(e2e, 3), "text": text, "wall": datetime.now().isoformat()}
            con.print(f"[bold green]#{n}[/] [{rec['utt_s']:.1f}s audio | asr {rec['asr_infer_s']*1e3:.0f} ms | "
                      f"end->text {e2e*1e3:.0f} ms]  {text}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.save_audio:
                sf.write(RESULTS_DIR / f"utt_{n:03d}.wav", audio, SR)

    threading.Thread(target=asr_worker, daemon=True).start()

    preroll = collections.deque(maxlen=int(args.preroll * SR / CHUNK) + 1)
    utt: list[np.ndarray] = []
    speaking = False
    t_on = t_last_speech = 0.0
    pending = np.zeros(0, dtype=np.float32)
    t0 = time.perf_counter()

    con.print(f"[bold]Listening[/] device={args.device} thr={args.threshold} hangover={args.min_silence}ms "
              f"preroll={args.preroll}s. Speak; Ctrl+C to stop.\n")
    def wav_feeder():
        """Push 32 ms chunks at real-time pace, then 2 s of silence so the last utterance closes."""
        from common import load_wav_16k

        a = np.concatenate([load_wav_16k(Path(args.wav)), np.zeros(2 * SR, dtype=np.float32)])
        t_begin = time.perf_counter()
        for i in range(0, len(a) - CHUNK + 1, CHUNK):
            target = t_begin + (i + CHUNK) / SR
            while time.perf_counter() < target:
                time.sleep(0.002)
            audio_q.put(a[i:i + CHUNK].copy())
        audio_q.put(None)

    import contextlib

    if args.wav:
        threading.Thread(target=wav_feeder, daemon=True).start()
        stream = contextlib.nullcontext()
        con.print(f"[bold]Simulated mic[/]: replaying {args.wav} in real time")
    else:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                                device=args.device, extra_settings=wasapi_settings(args.device), callback=mic_cb)
    try:
        with stream:
            while True:
                got = audio_q.get()
                if got is None:
                    raise KeyboardInterrupt
                pending = np.concatenate([pending, got])
                if args.duration and time.perf_counter() - t0 > args.duration:
                    raise KeyboardInterrupt
                while len(pending) >= CHUNK:
                    chunk, pending = pending[:CHUNK], pending[CHUNK:]
                    now = time.perf_counter()
                    p = vad(torch.from_numpy(chunk), SR).item()
                    if not speaking:
                        preroll.append(chunk)
                        if p >= args.threshold:
                            speaking, t_on, t_last_speech = True, now, now
                            utt = list(preroll)
                            sys.stdout.write(f"\r  [speech @ {now - t0:7.2f}s] ")
                            sys.stdout.flush()
                    else:
                        utt.append(chunk)
                        if p >= args.threshold:
                            t_last_speech = now
                        sil_ms = (now - t_last_speech) * 1e3
                        too_long = (now - t_on) >= args.max_utt
                        if sil_ms >= args.min_silence or too_long:
                            speaking = False
                            audio = np.concatenate(utt)
                            spoken_ms = (t_last_speech - t_on) * 1e3
                            if spoken_ms >= args.min_speech:
                                # t_last_speech is the moment the user actually stopped talking
                                asr_q.put((audio, t_last_speech, t_on, now))
                                sys.stdout.write(f"end (+{args.min_silence}ms hangover) -> ASR\n")
                            else:
                                sys.stdout.write(f"blip {spoken_ms:.0f}ms ignored\n")
                            preroll.clear()
                            vad.reset_states()
    except KeyboardInterrupt:
        pass
    finally:
        asr_q.put(None)
        time.sleep(1.0)
        if lat_e2e:
            con.print(f"\n[bold]Session summary[/] ({len(lat_e2e)} utterances)")
            con.print(f"  speech_end->text  p50 {pct(lat_e2e,50)*1e3:.0f} ms   p95 {pct(lat_e2e,95)*1e3:.0f} ms "
                      f"(includes the {args.min_silence} ms hangover)")
            con.print(f"  asr infer         p50 {pct(lat_asr,50)*1e3:.0f} ms   p95 {pct(lat_asr,95)*1e3:.0f} ms")
            con.print(f"  log -> {log_path}")


if __name__ == "__main__":
    main()
