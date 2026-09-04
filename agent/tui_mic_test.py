"""Headless test of the mic-mute button and F2 binding in the real Textual app.

    .venv\\Scripts\\python.exe agent\\tui_mic_test.py
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
from tui import AgentApp, Button, PromptArea, TextualUI  # noqa: E402


async def wait_for(pilot, cond, timeout=90):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        await pilot.pause(0.3)
        try:
            if cond():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


async def main() -> None:
    a = talk.build_parser().parse_args(["--no-mic", "--no-play", "--no-search"])
    ui = TextualUI(model=a.model, voice=a.voice)
    t = talk.Talk(a, ui=ui)

    def start():
        t.load()
        t.run(t.idle_chunks())

    app = AgentApp(ui, start, t.submit_text, mic_fn=t.toggle_mic)
    ui.app = app
    async with app.run_test(size=(110, 40)) as pilot:
        ok = await wait_for(pilot, lambda: any("ready" in str(w.render()) for w in app.query(".note")))
        print("agent ready:", "ok" if ok else "FAILED", file=sys.__stderr__, flush=True)

        btn = app.query_one("#mic_btn", Button)
        assert "MUTED" not in str(btn.label) and not btn.has_class("-muted"), "button should start unmuted"
        print("initial button state (unmuted):", "ok", file=sys.__stderr__, flush=True)

        await pilot.click("#mic_btn")
        clicked_ok = await wait_for(pilot, lambda: t.mic_muted is True)
        btn_ok = await wait_for(pilot, lambda: "MUTED" in str(btn.label) and btn.has_class("-muted"))
        print("click mutes talk.mic_muted:", "ok" if clicked_ok else "FAILED", file=sys.__stderr__, flush=True)
        print("click updates button label/style:", "ok" if btn_ok else "FAILED", file=sys.__stderr__, flush=True)

        await pilot.press("f2")
        f2_ok = await wait_for(pilot, lambda: t.mic_muted is False)
        btn_ok2 = await wait_for(pilot, lambda: "MUTED" not in str(btn.label) and not btn.has_class("-muted"))
        print("F2 unmutes talk.mic_muted:", "ok" if f2_ok else "FAILED", file=sys.__stderr__, flush=True)
        print("F2 restores button label/style:", "ok" if btn_ok2 else "FAILED", file=sys.__stderr__, flush=True)

        # focus should return to the editor after a click so typing keeps working
        inp = app.query_one("#input", PromptArea)
        await pilot.click("#mic_btn")
        await wait_for(pilot, lambda: t.mic_muted is True)
        assert app.focused is inp, f"focus did not return to the editor, got {app.focused}"
        print("focus returns to the editor after clicking the button:", "ok", file=sys.__stderr__, flush=True)

        t.quit.set()
    print("done", file=sys.__stderr__, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
