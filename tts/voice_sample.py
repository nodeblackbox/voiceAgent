"""Play a short line back to back in several Kokoro voices so you can pick one by ear.

Usage:
    python tts/voice_sample.py                                   # default lineup
    python tts/voice_sample.py --voices af_bella af_sky am_puck
    python tts/voice_sample.py --text "Testing, one two three."
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tts"))
from kokoro_stream import KokoroTTS, SpeakerSink

DEFAULT_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "af_nova", "am_michael", "am_puck", "bf_emma"]
LINE = "Hey, I'm your voice agent. This is what I sound like."


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
        raise SystemExit(f"no output device matching '{spec}'")
    return hits[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*", default=DEFAULT_VOICES)
    ap.add_argument("--out-device", default="Yeti")
    ap.add_argument("--text", default=LINE)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.5, help="seconds of silence between voices")
    args = ap.parse_args()

    print(f"loading Kokoro (voice={args.voices[0]}) ...")
    tts = KokoroTTS(voice=args.voices[0])
    sink = SpeakerSink(device=resolve_output_device(args.out_device))
    try:
        for v in args.voices:
            print(f"-- {v} --")
            tts.voice = v
            tts.pipeline.load_voice(v)
            audio = tts.synth(args.text, args.speed)
            sink.push(audio)
            sink.wait_drained(30)
            time.sleep(args.gap)
    finally:
        sink.close()
    print("done")


if __name__ == "__main__":
    main()
