"""Headless test of the Textual UI: paste a multi-line snippet, press Enter, check a paste block and an
agent block appear; then a /explain preset; then a plain question. No mic, no sound.

    .venv\\Scripts\\python.exe agent\\tui_test.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.argv = ["talk.py", "--no-mic", "--no-play", "--no-search", "--tui"]
import talk  # noqa: E402
from tui import AgentApp, PromptArea, TextualUI  # noqa: E402

SNIPPET = '''def barge_in(self):
    if self.sm.barge_in():
        self.cancel.set()
        if self.session:
            self.session.interrupt()
        self.ui.interrupted(*self._spoken_split())
        self.sm.resume_user()'''


async def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    # reuse talk's argument definitions by calling its main parser indirectly
    a = talk.build_parser().parse_args(["--no-mic", "--no-play", "--no-search"])
    a.tui = True
    ui = TextualUI(model=a.model, voice=a.voice)
    t = talk.Talk(a, ui=ui)

    def start():
        t.load()
        t.run(t.idle_chunks())

    app = AgentApp(ui, start, t.submit_text)
    ui.app = app
    async with app.run_test(size=(110, 40)) as pilot:
        # wait until the agent says it's ready (a note block containing 'ready')
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 90:
            await pilot.pause(0.5)
            if any("ready" in str(w.render()) for w in app.query(".note")):
                break
        print(f"ready after {time.perf_counter() - t0:.0f}s, notes: {[str(w.render())[:50] for w in app.query('.note')][:6]}", file=sys.__stderr__, flush=True)
        inp = app.query_one("#input", PromptArea)
        inp.insert(SNIPPET)              # what a paste does
        await pilot.pause(0.2)
        assert inp.text.count("\n") >= 6, "paste lost its lines"
        await pilot.press("enter")
        ok = await wait_for(app, pilot, lambda: app.query(".paste") and app.query(".agent"), 60)
        print("paste turn:", file=sys.__stderr__, flush=True) or print("", "ok" if ok else "FAILED")
        await wait_for(app, pilot, lambda: not t.sm.agent_busy and app._current is None, 90)
        inp.insert("/review")
        await pilot.press("enter")
        ok2 = await wait_for(app, pilot, lambda: len(app.query(".agent")) >= 2, 60)
        print("/review preset on last paste:", file=sys.__stderr__, flush=True) or print("", "ok" if ok2 else "FAILED")
        await wait_for(app, pilot, lambda: not t.sm.agent_busy and app._current is None, 90)
        inp.insert("what does sm stand for there")
        await pilot.press("shift+enter")
        inp.insert("one word answer")
        assert "\n" in inp.text, "shift+enter did not insert a newline"
        await pilot.press("enter")
        ok3 = await wait_for(app, pilot, lambda: len(app.query(".agent")) >= 3, 60)
        print("plain question with shift+enter newline:", file=sys.__stderr__, flush=True) or print("", "ok" if ok3 else "FAILED")
        await wait_for(app, pilot, lambda: not t.sm.agent_busy and app._current is None, 90)
        for w in app.query(".agent"):
            print("  agent block:", str(w.render())[:120].replace("\n", " "))
        t.quit.set()
    print("done", file=sys.__stderr__, flush=True)


async def wait_for(app, pilot, cond, timeout):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await pilot.pause(0.3)
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


if __name__ == "__main__":
    asyncio.run(main())
