"""Tests for the three bugs found in the 2026-09-06 live session:

  1. barge-in reset-to-zero on any single VAD dip (sustained real speech failed to cut; ducking/undocking
     flickered and spammed "backchannel ignored" during one continuous utterance)
  2. Smart Turn cutting mid-hedge on weak-confidence scores ("um", "so", "yeah yeah so...")
  3. the "heard up to: "..."" block rendering with nothing in it when barge-in happens before any words

These are algorithmic bugs, so they are verified with exact scripted sequences run through the REAL
`Talk.run()` loop (via the module's own `silero_vad.load_silero_vad`, scripted rather than mocked-out
logic) and the REAL `Endpointer.update()` (with the ONNX model swapped for a scripted score function so
the test runs in under a second, not a full round trip through Smart Turn/Parakeet/Kokoro).

`Talk.run()` blocks after its chunk generator is exhausted, waiting for a reply to finish — in these
synthetic scenarios nothing is really replying, so it's run in a daemon thread and simply left there;
the interesting state mutations all happen during chunk processing, which completes in milliseconds.

    .venv\\Scripts\\python.exe agent\\fixes_test.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bench"))
import talk  # noqa: E402
from endpoint import Endpointer  # noqa: E402
from state import S, StateMachine  # noqa: E402
from ui import TerminalUI  # noqa: E402

SR = talk.SR
CHUNK = talk.CHUNK
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok" if cond else "FAIL"
    print(f"{status}: {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_talk() -> talk.Talk:
    a = talk.build_parser().parse_args(["--no-play", "--no-search", "--plain", "--verbose"])
    t = talk.Talk.__new__(talk.Talk)
    t.a = a
    t.ui = TerminalUI(model=a.model, voice=a.voice, plain=True)
    t.ui.note = lambda *a_, **k: None  # asserted on internals, not printed output
    t.sm = StateMachine(on_change=lambda o, n, e: None)
    t.echo = MagicMock()
    t.echo.is_echo.return_value = False
    t.sink = MagicMock()
    t.sink.gain = 1.0
    t.session = None
    t.cancel = threading.Event()
    t.turn_q = talk.queue.Queue()
    t.asr = MagicMock()
    t.asr.recognize.return_value = ""
    t.asr_lock = threading.Lock()
    t.mic_muted = False
    t._need_reset = False
    t.vad = None
    t.backchannels = 0
    t.verbose = True
    t._last_partial_text = ""
    t.turns = []
    t.turn_n = 0
    t.quit = threading.Event()
    t.ep = Endpointer.__new__(Endpointer)
    t.ep.min_ms, t.ep.max_ms, t.ep.every_ms = 250, 1_000_000, 200
    t.ep.threshold, t.ep.high_threshold = 0.7, 0.9
    t.ep.model = lambda audio: 0.0
    t.ep.last_check_ms, t.ep.last_p, t.ep.checks, t.ep.held_on_filler, t.ep.cost_ms = -1e9, None, 0, 0, []
    return t


def drive(t: talk.Talk, scripted_vad: list[float], agent_state=S.SPEAKING, settle: float = 0.3) -> None:
    """Feeds len(scripted_vad) fake chunks through the real Talk.run(), one VAD score per chunk, in a
    daemon thread. run() blocks after the generator exhausts (waiting for a reply that never comes in
    this synthetic setup); that tail wait is irrelevant to what's being tested, so it is not joined —
    the daemon thread dies with the process. `settle` just gives the fast chunk-processing part time to
    finish before assertions run."""
    idx = {"i": 0}

    def vad_fn(chunk_tensor, sr):
        i = min(idx["i"], len(scripted_vad) - 1)
        idx["i"] += 1
        out = MagicMock()
        out.item.return_value = scripted_vad[i]
        return out
    vad_fn.reset_states = lambda: None

    import silero_vad
    silero_vad.load_silero_vad = lambda: vad_fn

    t.sm.state = agent_state
    gen = (np.zeros(CHUNK, dtype=np.float32) for _ in range(len(scripted_vad)))
    th = threading.Thread(target=t.run, args=(gen,), daemon=True)
    th.start()
    time.sleep(settle)


# --------------------------------------------------------------------------- 1a. sustained real speech with flutter
def test_sustained_speech_with_flutter():
    print("\n[1a] sustained real speech with natural VAD flutter (one dip every 6th chunk) must still cut")
    # 40 chunks of "speaking" at 0.85, but every 6th chunk dips to 0.4 (a plosive/sibilant misread) —
    # representative of real continuous speech, which is never uniformly >=threshold for 320ms straight.
    scores = [0.4 if i % 6 == 5 else 0.85 for i in range(40)]
    t = make_talk()
    drive(t, scores)
    check("real flutter-y sustained speech triggers a cut", any(e[1] == "barge_in" for e in t.sm.log),
          f"state log events: {[e[1] for e in t.sm.log]}")


def test_old_logic_would_have_failed():
    print("\n[1b] sanity check: the OLD hard-reset-on-any-miss logic really would have failed this case")
    scores = [0.4 if i % 6 == 5 else 0.85 for i in range(40)]
    hot, cut = 0, False
    for p in scores:
        if p >= 0.75:
            hot += 1
            if hot >= 10:
                cut = True
                break
        else:
            hot = 0  # the bug: one miss wipes all progress
    check("old logic never accumulates 10 consecutive hits with a dip every 6th chunk", not cut,
          "max run between dips is 5 chunks < bargein_chunks=10, so old logic could never reach it")


# --------------------------------------------------------------------------- 2. cough: duck only, one clean release
def test_cough_ducks_but_does_not_cut_and_fires_once():
    print("\n[2] a short cough-like burst ducks but never cuts, and 'backchannel ignored' fires exactly once")
    scores = [0.9] * 5 + [0.1] * 20  # 160 ms burst, then 640 ms of silence
    t = make_talk()
    notes = []
    t.ui.note = lambda msg, *a_, **k: notes.append(msg) if msg == "backchannel ignored" else None
    drive(t, scores)
    check("the cough never triggers a real cut", not any(e[1] == "barge_in" for e in t.sm.log))
    check("'backchannel ignored' fires exactly once, not repeatedly", len(notes) == 1, f"fired {len(notes)} times")
    check("backchannels counter incremented exactly once", t.backchannels == 1, f"backchannels={t.backchannels}")


def test_repeated_micro_pauses_do_not_spam_backchannel_notes():
    print("\n[3a] one utterance with several short internal pauses (the transcript's exact pattern — this has a "
          "67% duty cycle, i.e. is mostly continuous speech) must not spam 'backchannel ignored' on every micro-"
          "pause, even though it correctly escalates to a real interrupt within about a second")
    # mimics the transcript: speech, brief pause, speech, brief pause... — the OLD bug fired the note (and
    # flickered the volume) on every single one of these brief pauses; that is the thing being tested here,
    # not whether this particular high-duty-cycle pattern eventually cuts (it should, and does).
    scores = ([0.85] * 4 + [0.3] * 2) * 5  # 5 repeats, 30 chunks total
    t = make_talk()
    notes = []
    t.ui.note = lambda msg, *a_, **k: notes.append(msg) if msg == "backchannel ignored" else None
    drive(t, scores)
    check("it does escalate to a real interrupt (67% duty cycle is effectively continuous speech)",
          any(e[1] == "barge_in" for e in t.sm.log))
    check("but 'backchannel ignored' never spams along the way (old bug: once per micro-pause)", len(notes) == 0,
          f"fired {len(notes)} times: {notes}")

    print("\n[3b] three genuinely separate short bursts (real silence between them) each duck once, "
          "never cut, and each releases as exactly one clean 'backchannel ignored' — not zero, not a flicker")
    scores2 = ([0.85] * 4 + [0.1] * 10) * 3  # burst, real silence (well past release_chunks), repeat 3x
    t2 = make_talk()
    notes2 = []
    t2.ui.note = lambda msg, *a_, **k: notes2.append(msg) if msg == "backchannel ignored" else None
    drive(t2, scores2, settle=0.4)
    check("three separate short bursts never accumulate enough to cut", not any(e[1] == "barge_in" for e in t2.sm.log))
    check("each burst releases as exactly one note (3 bursts -> 3 notes, not 0 and not more)", len(notes2) == 3,
          f"fired {len(notes2)} times: {notes2}")
    check("backchannels counter matches", t2.backchannels == 3, f"backchannels={t2.backchannels}")


# --------------------------------------------------------------------------- 4. Smart Turn three-tier threshold
def test_smart_turn_tiers():
    print("\n[4] Endpointer: weak score + trailing filler word holds; same score with a clean ending cuts; "
          "a very high score cuts regardless of the trailing word")
    ep = Endpointer.__new__(Endpointer)
    ep.min_ms, ep.max_ms, ep.every_ms = 0, 1_000_000, 0
    ep.threshold, ep.high_threshold = 0.7, 0.9
    ep.last_check_ms, ep.last_p, ep.checks, ep.held_on_filler, ep.cost_ms = -1e9, None, 0, 0, []

    ep.model = lambda audio: 0.75
    ep.start_utterance()
    d = ep.update(np.zeros(1600, dtype=np.float32), 300, "so yeah I guess we're gonna")
    check("0.75 + a real trailing word -> not held (control)", d == "done", f"got {d!r}")

    ep.model = lambda audio: 0.75
    ep.start_utterance()
    d = ep.update(np.zeros(1600, dtype=np.float32), 300, "yeah, yeah, yeah so")
    check("0.75 + trailing filler word 'so' -> held, not done", d is None, f"got {d!r}")
    check("held_on_filler counter incremented", ep.held_on_filler == 1)

    ep.model = lambda audio: 0.95
    ep.start_utterance()
    d = ep.update(np.zeros(1600, dtype=np.float32), 300, "yeah, yeah, yeah so")
    check("0.95 (>= high_threshold) ends the turn even on a trailing filler word", d == "done", f"got {d!r}")

    ep.model = lambda audio: 0.4
    ep.start_utterance()
    d = ep.update(np.zeros(1600, dtype=np.float32), 300, "a complete sentence.")
    check("0.4 (below threshold) never ends the turn", d is None, f"got {d!r}")


# --------------------------------------------------------------------------- 5. empty-spoken interrupted() rendering
def test_empty_interrupted_rendering():
    print("\n[5] interrupting the agent before it has said anything renders cleanly, not 'heard up to: \"...\"'")
    ui = TerminalUI(model="m", voice="v", plain=True)
    lines: list[str] = []
    ui._line = lambda t: lines.append(str(t))
    ui.interrupted("", "some unspoken text")
    joined = " ".join(lines)
    check("no broken empty quote in the rendered output", "“…”" not in joined, repr(joined))
    check("says it was interrupted before anything was said", "before it said anything" in joined, repr(joined))

    from tui import TextualUI
    tui_ui = TextualUI(model="m", voice="v")
    captured = []
    tui_ui._call = lambda fn, *a_: captured.append(fn)
    tui_ui.interrupted("", "x")
    fake_app = MagicMock()
    added = []
    fake_app.add_block.side_effect = lambda text, cls, *a_, **k: added.append((str(text), cls))
    fake_app._current = None
    tui_ui.app = fake_app
    captured[-1]()
    check("Textual UI also avoids the broken empty quote", bool(added) and "“…”" not in added[-1][0], added)


def main() -> None:
    test_old_logic_would_have_failed()
    test_sustained_speech_with_flutter()
    test_cough_ducks_but_does_not_cut_and_fires_once()
    test_repeated_micro_pauses_do_not_spam_backchannel_notes()
    test_smart_turn_tiers()
    test_empty_interrupted_rendering()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
