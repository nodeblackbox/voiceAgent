"""Textual terminal UI for the voice agent: a real multi-line editor at the bottom, the conversation as
separate blocks above it, live state in the header.

Editor keys
    Enter            send (unless you are inside an unclosed ``` code fence)
    Shift+Enter      newline (Windows Terminal / kitty-protocol terminals); Ctrl+J and Alt+Enter always work
    Ctrl+Up/Down     previous / next thing you sent
    Ctrl+L           clear the editor          Ctrl+C / Ctrl+Q   quit
    F2 (or the button top-right)   mute/unmute the microphone — stops it listening, does not touch anything else
Pasting keeps every line (bracketed paste); nothing is sent until you press Enter.

The class `TextualUI` implements the same methods as ui.TerminalUI, so talk.py's loop is unchanged;
every call is marshalled onto the app thread with call_from_thread.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static, TextArea

STATE_STYLE = {
    "listening": "bold white on #4a5560",
    "user speaking": "bold black on #38bdf8",
    "thinking": "bold black on #facc15",
    "speaking": "bold black on #4ade80",
    "interrupted": "bold white on #ef4444",
}

CSS = """
Screen { layout: vertical; background: $background; }
#topbar { height: 1; background: $panel; }
#status { height: 1; padding: 0 1; width: 1fr; content-align: left middle; }
#mic_btn { min-width: 22; height: 1; padding: 0 1; margin: 0 1 0 0; border: none;
           background: #16a34a; color: #ffffff; text-style: bold; }
#mic_btn:hover { background: #15803d; }
#mic_btn.-muted { background: #dc2626; }
#mic_btn.-muted:hover { background: #b91c1c; }
#log { height: 1fr; padding: 0 1; }
.blk { margin: 0 0 1 0; padding: 0 1; }
.you { border-left: thick #38bdf8; color: #bae6fd; }
.agent { border-left: thick #4ade80; color: #dcfce7; }
.tool { border-left: thick #facc15; color: #fef3c7; margin: 0 0 0 2; }
.note { color: #94a3b8; margin: 0 0 0 2; }
.warn { color: #fbbf24; margin: 0 0 0 2; }
.err { color: #f87171; margin: 0 0 0 2; }
.cut { border-left: thick #ef4444; color: #fecaca; margin: 0 0 1 2; }
.paste { border-left: thick #a78bfa; color: #ddd6fe; }
.partial { color: #7dd3fc; text-style: italic; }
#meter { height: 1; padding: 0 1; color: #94a3b8; }
#input { height: auto; min-height: 3; max-height: 12; border: tall #7c3aed; background: $surface; }
#input:focus { border: tall #a78bfa; }
#hint { height: 1; padding: 0 1; color: #64748b; }
"""


class PromptArea(TextArea):
    """Multi-line editor: Enter sends, Shift+Enter / Ctrl+J / Alt+Enter insert a newline."""

    class Submit(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kw):
        super().__init__(soft_wrap=True, show_line_numbers=False, tab_behavior="indent", **kw)
        self._sent: deque[str] = deque(maxlen=100)
        self._hist_i = 0
        self._draft = ""

    def _inside_open_fence(self) -> bool:
        return self.text.count("```") % 2 == 1

    async def _on_key(self, event: events.Key) -> None:
        k = event.key
        if k == "enter":
            if self._inside_open_fence():
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return
            event.prevent_default()
            event.stop()
            text = self.text.rstrip("\n")
            if text.strip():
                self._sent.append(text)
            self._hist_i = len(self._sent)
            self.clear()
            if text.strip():
                self.post_message(self.Submit(text))
            return
        if k in ("shift+enter", "ctrl+j", "alt+enter", "escape+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if k == "ctrl+up" and self._sent:
            event.prevent_default()
            event.stop()
            if self._hist_i == len(self._sent):
                self._draft = self.text
            self._hist_i = max(0, self._hist_i - 1)
            self.load_text(self._sent[self._hist_i])
            return
        if k == "ctrl+down" and self._sent:
            event.prevent_default()
            event.stop()
            self._hist_i = min(len(self._sent), self._hist_i + 1)
            self.load_text(self._sent[self._hist_i] if self._hist_i < len(self._sent) else self._draft)
            return
        if k == "ctrl+l":
            event.prevent_default()
            event.stop()
            self.clear()
            return
        await super()._on_key(event)


class AgentApp(App):
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+q", "quit", "quit", show=False),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("f2", "toggle_mic", "mic on/off", show=True),
    ]

    def __init__(self, ui: "TextualUI", start_fn, submit_fn, mic_fn=None):
        super().__init__()
        self.ui = ui
        self.start_fn = start_fn      # runs the agent (blocking) in a thread once the UI is up
        self.submit_fn = submit_fn    # called with the editor text
        self.mic_fn = mic_fn          # called (no args) to toggle mic mute
        self._current: Static | None = None
        self._current_text = ""
        self._dirty = False
        self._partial: Static | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("", id="status")
            yield Button("🎤 mic on  ·  F2", id="mic_btn")
        yield VerticalScroll(id="log")
        yield Static("", id="meter")
        yield PromptArea(id="input")
        yield Static(" Enter send · Shift+Enter / Ctrl+J newline · Ctrl+↑↓ history · F2 mic on/off · "
                     "/help · /explain /review /next after a paste", id="hint")

    def on_mount(self) -> None:
        self.query_one("#input", PromptArea).focus()
        self.set_interval(1 / 15, self._tick)
        self.render_status()
        threading.Thread(target=self._runner, daemon=True, name="agent").start()

    def _runner(self):
        try:
            self.start_fn()
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.add_block, f"agent loop crashed: {type(e).__name__}: {e}", "err")
        finally:
            self.call_from_thread(self.add_block, "agent stopped · Ctrl+Q to exit", "warn")

    # ---------------------------------------------------------------- input
    def on_prompt_area_submit(self, msg: PromptArea.Submit) -> None:
        text = msg.text
        self.run_worker(lambda: self.submit_fn(text), thread=True, exclusive=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mic_btn" and self.mic_fn:
            self.run_worker(lambda: self.mic_fn(), thread=True, exclusive=False)
            self.query_one("#input", PromptArea).focus()

    def action_toggle_mic(self) -> None:
        if self.mic_fn:
            self.run_worker(lambda: self.mic_fn(), thread=True, exclusive=False)

    # ---------------------------------------------------------------- rendering helpers (app thread only)
    def render_status(self):
        ui = self.ui
        st = Text(f" {ui.state.upper()} ", style=STATE_STYLE.get(ui.state, "bold"))
        line = Text.assemble(st, "  ", (ui.model, "bold"), "  ", (f"voice {ui.voice}", "dim"), "   ",
                             ("  ".join(f"{k} {v}" for k, v in ui.badges.items()), "magenta"),
                             ("   " + ui.status if ui.status else "", "yellow"))
        self.query_one("#status", Static).update(line)
        self.render_mic()

    def render_mic(self):
        btn = self.query_one("#mic_btn", Button)
        muted = self.ui.mic_muted
        btn.label = "🔇 MIC MUTED  ·  F2" if muted else "🎤 mic on  ·  F2"
        btn.set_class(muted, "-muted")

    def render_meter(self):
        ui = self.ui
        bar = "█" * int(ui.vad_p * 20)
        self.query_one("#meter", Static).update(Text.assemble(
            ("vad ", "dim"), (f"{bar:<20}", "#38bdf8" if ui.vad_p >= 0.5 else "#475569"), (f" {ui.vad_p:4.2f}", "dim"),
            ("   echo ", "dim"), (f"{ui.echo:4.2f}", "#f87171" if ui.echo > 0.45 else "dim"),
            ("   ", ""), (ui.last_transition, "dim")))

    def add_block(self, text: str | Text, cls: str, title: str | None = None) -> Static:
        w = Static(text, classes=f"blk {cls}")
        if title:
            w.border_title = title
        log = self.query_one("#log", VerticalScroll)
        log.mount(w)
        log.scroll_end(animate=False)
        return w

    def _tick(self):
        if self._dirty and self._current is not None:
            self._current.update(self._current_text)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
            self._dirty = False
        self.render_meter()


class TextualUI:
    """Same surface as ui.TerminalUI. Thread-safe: every method may be called from any thread."""

    def __init__(self, model: str, voice: str, **_):
        self.model, self.voice = model, voice
        self.state = "listening"
        self.mic_muted = False
        self.badges: dict[str, str] = {}
        self.status = ""
        self.vad_p = 0.0
        self.echo = 0.0
        self.last_transition = ""
        self.t0 = time.perf_counter()
        self.app: AgentApp | None = None
        self.plain = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _call(self, fn, *args):
        app = self.app
        if app is None or not getattr(app, "is_running", False) or getattr(app, "_exit", False):
            return
        try:
            app.call_from_thread(fn, *args)
        except Exception:  # noqa: BLE001  (app shutting down)
            pass

    # --- state / meters ------------------------------------------------------------------------
    def set_state(self, state: str, transition: str = ""):
        self.state = state
        if transition:
            self.last_transition = f"{time.perf_counter() - self.t0:7.1f}s  {transition}"
        self._call(self.app.render_status if self.app else (lambda: None))

    def set_vad(self, p: float, echo: float = 0.0):
        self.vad_p, self.echo = p, echo

    def set_status(self, msg: str):
        self.status = msg
        self._call(self.app.render_status if self.app else (lambda: None))

    def set_input(self, buf: str):
        pass

    def set_mic(self, muted: bool):
        self.mic_muted = muted
        self._call(self.app.render_mic if self.app else (lambda: None))

    # --- conversation blocks -------------------------------------------------------------------
    def set_partial(self, text: str):
        def go():
            if self.app._partial is None:
                self.app._partial = self.app.add_block("", "partial")
            self.app._partial.update(Text(f"… {text} ▍", style="italic"))
            self.app.query_one("#log", VerticalScroll).scroll_end(animate=False)
        self._call(go)

    def user(self, n: int, text: str, asr_ms: float | None = None):
        def go():
            if self.app._partial is not None:
                self.app._partial.remove()
                self.app._partial = None
            title = f"#{n} you" + (f" · asr {asr_ms:.0f} ms" if asr_ms else "")
            self.app.add_block(text, "you", title)
        self._call(go)

    def paste(self, n: int, head: str, lines: int, instruction: str):
        def go():
            body = Text.assemble((instruction + "\n", "bold"), (f"{head}\n", ""), (f"… {lines} lines pasted", "dim"))
            self.app.add_block(body, "paste", f"#{n} you · pasted")
        self._call(go)

    def assistant_start(self, n: int):
        def go():
            self.app._current_text = ""
            self.app._current = self.app.add_block("", "agent", f"#{n} {self.voice}")
        self._call(go)

    def assistant_text(self, delta: str):
        def go():
            self.app._current_text += delta
            self.app._dirty = True
        self._call(go)

    def assistant_done(self, badges: dict[str, str]):
        def go():
            self.badges = badges
            if self.app._current is not None:
                self.app._current.update(self.app._current_text)
                self.app._current.border_subtitle = "  ".join(f"{k} {v}" for k, v in badges.items())
            self.app._current = None
            self.app.render_status()
        self._call(go)

    def tool(self, name: str, result: str | None = None):
        txt = f"⚙ {name}(...)" if result is None else f"⚙ {name} → {result[:110]}"
        self._call(self.app.add_block if self.app else (lambda *a: None), txt, "tool")

    def interrupted(self, spoken: str, unspoken: str):
        def go():
            if self.app._current is not None:
                self.app._current.update(self.app._current_text)
                self.app._current.border_subtitle = "✂ interrupted"
                self.app._current = None
            if spoken.strip():
                t = Text.assemble(("heard up to: ", "dim"), (f"“…{spoken[-70:]}”", ""),
                                  (f"   dropped {len(unspoken.split())} words" if unspoken else "", "dim"))
            else:
                t = Text("interrupted before it said anything", style="dim")
            self.app.add_block(t, "cut")
        self._call(go)

    def note(self, msg: str, style: str = "dim"):
        cls = "err" if "red" in style else "warn" if "yellow" in style else "note"
        self._call(self.app.add_block if self.app else (lambda *a: None), msg, cls)


def run_app(ui: TextualUI, start_fn, submit_fn, mic_fn=None) -> None:
    app = AgentApp(ui, start_fn, submit_fn, mic_fn=mic_fn)
    ui.app = app
    app.run()
