"""Echo agent (temporary, no LLM yet): listen with Silero VAD, transcribe with Parakeet the moment
you stop talking, and speak your own words back immediately with Kokoro (voice af_heart by default).

Both models load once at startup and stay resident in this process for its whole life; every number
printed per turn is the real repeat-call cost (no reload, no warm-up hiding in there).

The number that matters is "speech-end -> speaking": the gap between when you stop talking and when
audio actually starts coming out of the speaker. That's ASR time + TTS gen time for the first
sentence, nothing else.

Usage:
    python agent/echo_agent.py                          # Yeti mic in, Yeti headphones out, af_heart
    python agent/echo_agent.py --voice af_bella --out-device "BenQ"
    python agent/echo_agent.py --duration 60             # auto-stop instead of Ctrl+C
Ctrl+C to stop.
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
import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tts"))
from common import RESULTS_DIR, assert_cuda_still_active, load_parakeet, resolve_input_device, wasapi_settings, pct
from kokoro_stream import SR as TTS_SR, KokoroTTS, SpeakerSink

con = Console()
SR = 16000
CHUNK = 512  # 32 ms


def resolve_output_device(spec):
    import sounddevice as sd

    if spec is None:
        return None
    if str(spec).isdigit():
        return int(spec)
    apis = sd.query_hostapis()
    hits = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower()]
    hits.sort(key=lambda t: 0 if "WASAPI" in apis[t[1]["hostapi"]]["name"] else 1)
    if not hits:
        raise SystemExit(f"no output device matching '{spec}'. Run bench/list_devices.py")
    idx = hits[0][0]
    d = sd.query_devices(idx)
    con.print(f"output device {idx}: {d['name']} [{apis[d['hostapi']]['name']}]")
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-device", default="Yeti", help="mic: name substring or index")
    ap.add_argument("--out-device", default="Yeti", help="speaker: name substring or index")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-silence", type=int, default=600, help="hangover ms of silence that ends an utterance")
    ap.add_argument("--min-speech", type=int, default=250, help="ignore blips shorter than this (ms)")
    ap.add_argument("--preroll", type=float, default=1.0, help="seconds of audio kept before VAD trigger")
    ap.add_argument("--max-utt", type=float, default=30.0, help="hard cap: force-cut utterance after N s")
    ap.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = until Ctrl+C)")
    args = ap.parse_args()

    in_dev = resolve_input_device(args.in_device)
    out_dev = resolve_output_device(args.out_device)

    from silero_vad import load_silero_vad

    vad = load_silero_vad()

    con.print("[bold]Loading Parakeet[/] (hybrid: CUDA encoder + int8 CPU decoder) ...")
    asr, load_s, active = load_parakeet("hybrid")
    con.print(f"  loaded in {load_s:.1f}s providers={active}")
    asr.recognize(np.zeros(SR, dtype=np.float32))  # warm-up
    con.print(f"  providers after warm-up: {assert_cuda_still_active(asr, 'hybrid')}")

    con.print(f"[bold]Loading Kokoro[/] voice={args.voice} ...")
    tts = KokoroTTS(voice=args.voice)
    con.print(f"  loaded in {tts.load_s:.1f}s warm-up {tts.warmup_s:.2f}s device={tts.device} "
              f"VRAM {tts.vram_mib():.0f} MiB (torch allocated)")

    sink = SpeakerSink(device=out_dev)

    log_path = RESULTS_DIR / f"echo_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    turn_q: queue.Queue[tuple] = queue.Queue()
    lat_e2e, lat_asr, lat_tts = [], [], []

    def mic_cb(indata, frames, t, status):
        if status:
            con.print(f"[red]mic: {status}[/]")
        audio_q.put(indata[:, 0].copy())

    def worker() -> None:
        n = 0
        while True:
            item = turn_q.get()
            if item is None:
                return
            n += 1
            audio, t_speech_end = item
            t0 = time.perf_counter()
            text = asr.recognize(audio)
            asr_s = time.perf_counter() - t0
            if not text.strip():
                con.print(f"[dim]#{n} (no speech recognized, {len(audio)/SR:.1f}s audio)[/]")
                continue

            played_before = sink.played_samples
            t0 = time.perf_counter()
            reply_audio = tts.synth(text, args.speed)
            tts_s = time.perf_counter() - t0
            sink.push(reply_audio)

            deadline = time.perf_counter() + 5
            while sink.played_samples <= played_before and time.perf_counter() < deadline:
                time.sleep(0.001)
            e2e_s = time.perf_counter() - t_speech_end

            lat_e2e.append(e2e_s)
            lat_asr.append(asr_s)
            lat_tts.append(tts_s)
            rec = {"n": n, "utt_s": round(len(audio) / SR, 2), "text": text,
                   "asr_ms": round(asr_s * 1e3, 1), "tts_gen_ms": round(tts_s * 1e3, 1),
                   "reply_s": round(len(reply_audio) / TTS_SR, 2),
                   "speech_end_to_speaking_ms": round(e2e_s * 1e3, 1), "wall": datetime.now().isoformat()}
            con.print(f"[bold green]#{n}[/] heard: \"{text}\"")
            con.print(f"     asr {rec['asr_ms']:.0f} ms | tts gen {rec['tts_gen_ms']:.0f} ms | "
                      f"reply {rec['reply_s']:.1f}s | [bold]speech-end -> speaking {rec['speech_end_to_speaking_ms']:.0f} ms[/]")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    threading.Thread(target=worker, daemon=True).start()

    preroll = collections.deque(maxlen=int(args.preroll * SR / CHUNK) + 1)
    utt: list[np.ndarray] = []
    speaking = False
    t_on = t_last_speech = 0.0
    pending = np.zeros(0, dtype=np.float32)
    t0 = time.perf_counter()

    stream = None
    try:
        import sounddevice as sd

        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK,
                                 device=in_dev, extra_settings=wasapi_settings(in_dev), callback=mic_cb)
        con.print(f"\n[bold]Listening[/] thr={args.threshold} hangover={args.min_silence}ms "
                  f"preroll={args.preroll}s. Speak; Ctrl+C to stop.\n")
        with stream:
            while True:
                got = audio_q.get()
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
                                turn_q.put((audio, t_last_speech))
                                sys.stdout.write(f"end (+{args.min_silence}ms hangover) -> ASR+TTS\n")
                            else:
                                sys.stdout.write(f"blip {spoken_ms:.0f}ms ignored\n")
                            preroll.clear()
                            vad.reset_states()
    except KeyboardInterrupt:
        pass
    finally:
        turn_q.put(None)
        time.sleep(0.5)
        sink.wait_drained(10)
        sink.close()
        if lat_e2e:
            con.print(f"\n[bold]Session summary[/] ({len(lat_e2e)} turns)")
            con.print(f"  speech-end -> speaking   p50 {pct(lat_e2e,50)*1e3:.0f} ms   p95 {pct(lat_e2e,95)*1e3:.0f} ms")
            con.print(f"  asr                      p50 {pct(lat_asr,50)*1e3:.0f} ms   p95 {pct(lat_asr,95)*1e3:.0f} ms")
            con.print(f"  tts gen                  p50 {pct(lat_tts,50)*1e3:.0f} ms   p95 {pct(lat_tts,95)*1e3:.0f} ms")
            con.print(f"  log -> {log_path}")


if __name__ == "__main__":
    main()
