"""Real-audio, real-loop test of mic mute: mutes, plays an unrelated clip (must stay muted, no turn),
plays a real synthesized "Hey Yeti" clip (must un-mute + ack), then plays the unrelated clip again
(must now become a real turn). No mocks below Talk itself — real Silero, real Parakeet, real state machine.

    .venv\\Scripts\\python.exe agent\\mic_mute_sim_test.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "bench"))
import talk  # noqa: E402
from common import load_wav_16k  # noqa: E402

SR = 16000
CHUNK = 512


def feed(t: talk.Talk, path: Path, realtime: bool = True):
    audio = load_wav_16k(path)
    audio = np.concatenate([audio, np.zeros(int(SR * 1.0), dtype=np.float32)])  # trailing silence to close the utterance
    for i in range(0, len(audio) - CHUNK, CHUNK):
        yield audio[i:i + CHUNK]
        if realtime:
            time.sleep(CHUNK / SR)


def main() -> None:
    a = talk.build_parser().parse_args(["--no-play", "--no-search", "--plain", "--min-silence", "250", "--wake-hangover", "400"])
    t = talk.Talk(a)
    events = []
    orig_note = t.ui.note
    t.ui.note = lambda msg, style="dim": (events.append(msg), orig_note(msg, style))[1]

    def chunks():
        yield from silence(0.3)
        print("[test] muting mic", file=sys.__stderr__)
        t.toggle_mic(True)
        assert t.mic_muted, "toggle_mic(True) should mute immediately"
        yield from silence(0.3)
        print("[test] feeding OFF-TOPIC clip while muted (must NOT unmute, must NOT become a turn)", file=sys.__stderr__)
        yield from feed(t, ROOT / "audio" / "wake_test_offtopic.wav")
        yield from silence(1.0)
        assert t.mic_muted, "an unrelated utterance must not unmute the mic"
        assert t.turn_n == 0, f"an unrelated utterance while muted must not become a turn, got turn_n={t.turn_n}"
        print("[test] OK: stayed muted, no turn created", file=sys.__stderr__)

        print("[test] feeding the WAKE clip ('Hey Yeti, are you there?')", file=sys.__stderr__)
        yield from feed(t, ROOT / "audio" / "wake_test_hey_yeti.wav")
        deadline = time.perf_counter() + 15
        while t.mic_muted and time.perf_counter() < deadline:
            yield np.zeros(CHUNK, dtype=np.float32)
            time.sleep(CHUNK / SR)
        assert not t.mic_muted, "the wake phrase must unmute the mic"
        print("[test] OK: wake phrase unmuted the mic", file=sys.__stderr__)
        yield from silence(1.5)  # let the "Yeah?" ack finish playing / turn settle

        print("[test] feeding the OFF-TOPIC clip again, now unmuted (must become a real turn)", file=sys.__stderr__)
        yield from feed(t, ROOT / "audio" / "wake_test_offtopic.wav")
        deadline = time.perf_counter() + 30
        while not t.turns and time.perf_counter() < deadline:
            yield np.zeros(CHUNK, dtype=np.float32)
            time.sleep(CHUNK / SR)
        assert t.turns, "once unmuted, a real utterance must become a completed turn"
        print(f"[test] OK: turn #{t.turn_n} created and completed: {t.turns[0].user_text!r}", file=sys.__stderr__)
        t.quit.set()

    def silence(seconds: float):
        for _ in range(int(seconds * SR / CHUNK)):
            yield np.zeros(CHUNK, dtype=np.float32)
            time.sleep(CHUNK / SR)

    print("loading models…", file=sys.__stderr__)
    t.load()
    t.run(chunks())
    print(f"[test] wake_words={t.wake_words} mic_muted_final={t.mic_muted} turns={t.turn_n}", file=sys.__stderr__)
    print("[test] ALL ASSERTIONS PASSED", file=sys.__stderr__)


if __name__ == "__main__":
    try:
        main()
        print("DONE OK", file=sys.__stderr__)
    except AssertionError as e:
        print(f"TEST FAILED: {e}", file=sys.__stderr__)
        raise
