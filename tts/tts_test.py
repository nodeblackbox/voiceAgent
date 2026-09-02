"""Kokoro streaming TTS benchmark.

  1. offline: per-sentence generation time, RTF, VRAM
  2. stream:  simulate an LLM emitting words at --wps words/s; measure time-to-first-audio and gaps
  3. barge-in: interrupt mid-utterance and measure how fast audio actually stops
  4. round trip: Parakeet transcribes what Kokoro said; WER against the input text

Usage:
  python tts/tts_test.py                # NullSink (no sound), all four experiments
  python tts/tts_test.py --play         # through the default output device (Yeti headphone jack)
  python tts/tts_test.py --play --out "BenQ" --voice af_bella --wps 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tts"))
sys.path.insert(0, str(ROOT / "bench"))
from kokoro_stream import SR, KokoroTTS, NullSink, SentenceChunker, SpeakerSink, StreamingSpeech

con = Console()
RESULTS = ROOT / "results"

TEXT = (
    "Okay, I looked at the limit order problem. The order does get placed, but the chart never draws "
    "the line for it, so you can't see where it sits. I think the fix is to draw the line the moment "
    "the order is accepted, and let you drag it to a new price. That would take about 45 minutes. "
    "Do you want me to do that now, or after the barge-in test?"
)


def resolve_out(spec):
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
        raise SystemExit(f"no output device matching {spec}")
    return hits[0][0]


def make_sink(args):
    if args.play:
        dev = resolve_out(args.out)
        import sounddevice as sd

        d = sd.query_devices(dev if dev is not None else sd.default.device[1])
        con.print(f"playing on: {d['name']}")
        return SpeakerSink(device=dev)
    return NullSink()


def exp_offline(tts: KokoroTTS, text: str, speed: float) -> dict:
    con.print("\n[bold]1. Offline: one Kokoro call per sentence[/]")
    sentences = SentenceChunker(first_min_words=0).feed(text + " ")
    rows = []
    t = Table()
    for c in ["#", "chars", "gen ms", "audio s", "RTF", "sentence"]:
        t.add_column(c, justify="right" if c != "sentence" else "left")
    for i, s in enumerate(sentences):
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            a = tts.synth(s, speed)
            torch.cuda.synchronize() if tts.device == "cuda" else None
            best = min(best, time.perf_counter() - t0)
        dur = len(a) / SR
        rows.append({"i": i, "chars": len(s), "gen_ms": round(best * 1e3, 1), "audio_s": round(dur, 2),
                     "rtf": round(best / dur, 3), "text": s})
        t.add_row(str(i), str(len(s)), f"{best*1e3:.0f}", f"{dur:.2f}", f"{best/dur:.3f}", s[:60])
    con.print(t)
    return {"rows": rows, "vram_mib": tts.vram_mib()}


def exp_stream(tts, args, text, label="stream") -> dict:
    con.print(f"\n[bold]2. Streaming: words arrive at {args.wps} words/s (simulated LLM)[/]")
    sink = make_sink(args)
    sess = StreamingSpeech(tts, sink, speed=args.speed)
    words = text.split(" ")
    t0 = time.perf_counter()
    for k, w in enumerate(words):
        sess.feed(w + (" " if k < len(words) - 1 else ""))
        time.sleep(1.0 / args.wps)
    sess.end_of_text()
    m = sess.wait()
    sink.close()
    con.print(f"  first text -> first audio at speaker: [bold]{m.ttfa_ms:.0f} ms[/]" if m.ttfa_ms else "  no audio")
    con.print(f"  text finished at {m.t_last_text:.2f}s, audio finished at {m.t_done:.2f}s, "
              f"gaps/underruns after start: {m.underruns}")
    t = Table()
    for c in ["#", "ready s", "gen ms", "audio s", "RTF", "sentence"]:
        t.add_column(c, justify="right" if c != "sentence" else "left")
    for s in m.sentences:
        t.add_row(str(s.idx), f"{s.t_ready:.2f}", f"{s.gen_s*1e3:.0f}", f"{s.audio_s:.2f}", f"{s.rtf:.3f}", s.text[:60])
    con.print(t)
    return {"ttfa_ms": m.ttfa_ms, "t_last_text": m.t_last_text, "t_done": m.t_done, "underruns": m.underruns,
            "sentences": [s.__dict__ for s in m.sentences]}


def exp_bargein(tts, args, text) -> dict:
    con.print("\n[bold]3. Barge-in: interrupt 1.5 s after first audio[/]")
    sink = make_sink(args)
    sess = StreamingSpeech(tts, sink, speed=args.speed)
    sess.feed(text)
    sess.end_of_text()
    while sink.first_audio_wall is None:
        time.sleep(0.005)
    time.sleep(1.5)
    played_before = sink.played_samples
    t_int = time.perf_counter()
    sess.interrupt()
    # how long until the speaker stops emitting new samples?
    last = sink.played_samples
    t_stop = None
    for _ in range(200):
        time.sleep(0.005)
        if sink.played_samples == last and sink.queued_samples() == 0:
            t_stop = time.perf_counter()
            break
        last = sink.played_samples
    extra_ms = (sink.played_samples - played_before) / SR * 1e3
    stop_ms = (t_stop - t_int) * 1e3 if t_stop else None
    sess.wait(5)
    sink.close()
    con.print(f"  audio emitted after interrupt(): {extra_ms:.0f} ms of samples; stopped within "
              f"{stop_ms:.0f} ms" if stop_ms else f"  extra {extra_ms:.0f} ms")
    return {"extra_audio_ms": extra_ms, "stop_ms": stop_ms}


def exp_roundtrip(tts, text, speed) -> dict:
    con.print("\n[bold]4. Round trip: Parakeet listens to Kokoro[/]")
    audio = tts.synth(text, speed)
    wav = RESULTS / "tts_roundtrip.wav"
    sf.write(wav, audio, SR)
    from common import load_parakeet, wer

    asr, _, _ = load_parakeet("hybrid")
    # Parakeet wants 16 kHz
    idx = np.arange(0, len(audio), SR / 16000)
    a16 = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    t0 = time.perf_counter()
    hyp = asr.recognize(a16)
    asr_s = time.perf_counter() - t0
    w = wer(text, hyp)
    con.print(f"  Kokoro said {len(audio)/SR:.1f} s of audio; Parakeet took {asr_s*1e3:.0f} ms; WER {w:.3f}")
    con.print(f"  heard: {hyp}")
    return {"wer": w, "asr_ms": asr_s * 1e3, "audio_s": len(audio) / SR, "hyp": hyp, "wav": str(wav)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wps", type=float, default=25, help="simulated LLM words per second")
    ap.add_argument("--play", action="store_true", help="actually play through the speakers")
    ap.add_argument("--out", default=None, help="output device name substring or index (with --play)")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--skip", nargs="*", default=[], choices=["offline", "stream", "bargein", "roundtrip"])
    args = ap.parse_args()

    before = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    tts = KokoroTTS(voice=args.voice)
    con.print(f"[bold]Kokoro[/] voice={args.voice} device={tts.device} load {tts.load_s:.1f}s warm-up {tts.warmup_s:.2f}s "
              f"VRAM {tts.vram_mib():.0f} MiB (torch allocated)")
    out = {"voice": args.voice, "speed": args.speed, "load_s": tts.load_s, "warmup_s": tts.warmup_s,
           "vram_mib": tts.vram_mib(), "wps": args.wps, "play": args.play}
    if "offline" not in args.skip:
        out["offline"] = exp_offline(tts, args.text, args.speed)
    if "stream" not in args.skip:
        out["stream"] = exp_stream(tts, args, args.text)
    if "bargein" not in args.skip:
        out["bargein"] = exp_bargein(tts, args, args.text)
    if "roundtrip" not in args.skip:
        out["roundtrip"] = exp_roundtrip(tts, args.text, args.speed)
    p = RESULTS / f"tts_{args.voice}{'_play' if args.play else ''}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    con.print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
