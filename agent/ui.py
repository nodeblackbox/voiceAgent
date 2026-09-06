"""Terminal UI for the voice agent (Rich Live). Everything the loop knows is on screen:

  header   state (colour-coded), model, voice, per-turn latency badges
  body     the conversation: your words (cyan, with the live partial while you talk), the agent's
           words streaming in (green), tool calls (yellow), interruptions (red rule)
  footer   VAD meter, echo-gate reading, last state transition

Never print() while this is live; go through TerminalUI methods.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATE_STYLE = {
    "listening": "bold white on grey35",
    "user speaking": "bold black on cyan",
    "thinking": "bold black on yellow",
    "speaking": "bold black on green",
    "interrupted": "bold white on red",
}


class TerminalUI:
    def __init__(self, model: str, voice: str, max_lines: int = 40, plain: bool = False):
        self.plain = plain  # plain: no Live screen, one line per event (for logs / non-interactive runs)
        self.console = Console(force_terminal=not plain, legacy_windows=False)
        self.model, self.voice = model, voice
        self.state = "listening"
        self.mic_muted = False
        self.partial = ""
        self.entries: deque[Text] = deque(maxlen=max_lines)
        self.current: Text | None = None    # the assistant line being streamed
        self.vad_p = 0.0
        self.echo = 0.0
        self.last_transition = ""
        self.badges: dict[str, str] = {}
        self.status = ""
        self.input_buf = ""
        self.lock = threading.Lock()
        self.live: Live | None = None
        self.t0 = time.perf_counter()

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self):
        if not self.plain:
            self.live = Live(self.render(), console=self.console, refresh_per_second=15, transient=False)
            self.live.__enter__()
        return self

    def __exit__(self, *a):
        if self.live:
            self.live.update(self.render(), refresh=True)
            self.live.__exit__(*a)

    def refresh(self):
        if self.live:
            self.live.update(self.render())

    def _line(self, text: Text):
        if self.plain:
            self.console.print(text)

    # ---------------------------------------------------------------- events
    def set_state(self, state: str, transition: str = ""):
        with self.lock:
            self.state = state
            if transition:
                self.last_transition = f"{time.perf_counter() - self.t0:7.2f}s  {transition}"
        if self.plain and transition:
            self._line(Text(f"[{time.perf_counter() - self.t0:7.2f}s] {transition}", style="dim"))
        self.refresh()

    def set_vad(self, p: float, echo: float = 0.0):
        self.vad_p, self.echo = p, echo


    def user(self, n: int, text: str, asr_ms: float | None = None):
        with self.lock:
            self.partial = ""
            t = Text()
            t.append(f"#{n} you  ", style="bold cyan")
            t.append(text, style="cyan")
            if asr_ms is not None:
                t.append(f"   ·{asr_ms:.0f} ms asr", style="dim")
            self.entries.append(t)
        self._line(t)
        self.refresh()

    def assistant_start(self, n: int):
        with self.lock:
            self.current = Text()
            self.current.append(f"#{n} {self.voice}  ", style="bold green")
            self.entries.append(self.current)
        self.refresh()

    def assistant_text(self, delta: str):
        with self.lock:
            if self.current is not None:
                self.current.append(delta, style="green")
        self.refresh()

    def assistant_done(self, badges: dict[str, str]):
        with self.lock:
            self.badges = badges
            cur = self.current
            self.current = None
        if cur is not None:
            self._line(cur)
            self._line(Text("     " + "  ".join(f"{k} {v}" for k, v in badges.items()), style="magenta"))
        self.refresh()

    def tool(self, name: str, result: str | None = None):
        with self.lock:
            t = Text()
            if result is None:
                t.append(f"     ⚙ {name}(...)", style="yellow")
            else:
                t.append(f"     ⚙ {name} → {result[:90]}", style="yellow dim")
            self.entries.append(t)
        self._line(t)
        self.refresh()

    def interrupted(self, spoken: str, unspoken: str):
        with self.lock:
            cur = self.current
            if cur is not None:
                cur.append("  ✂ interrupted", style="bold red")
                self.current = None
        if cur is not None:
            self._line(cur)
        with self.lock:
            t = Text()
            if spoken.strip():
                t.append("     heard up to: ", style="red dim")
                t.append(f"“…{spoken[-60:]}”", style="red")
                if unspoken:
                    t.append(f"   dropped {len(unspoken.split())} words", style="red dim")
            else:
                t.append("     interrupted before it said anything", style="red dim")
            self.entries.append(t)
        self._line(Text("     ✂ interrupted", style="bold red"))
        self._line(t)
        self.refresh()

    def note(self, msg: str, style: str = "dim"):
        with self.lock:
            t = Text(f"     {msg}", style=style)
            self.entries.append(t)
        self._line(t)
        self.refresh()

    def set_partial(self, text: str):
        with self.lock:
            self.partial = text
        if self.plain and text:
            self._line(Text(f"   … {text}", style="italic cyan dim"))
        self.refresh()

    def set_status(self, msg: str):
        self.status = msg
        self.refresh()

    def set_input(self, buf: str):
        self.input_buf = buf
        self.refresh()

    def set_mic(self, muted: bool):
        self.mic_muted = muted
        self.refresh()

    # ---------------------------------------------------------------- render
    def render(self):
        head = Table.grid(expand=True)
        head.add_column(ratio=1)
        head.add_column(justify="right")
        st = Text(f" {self.state.upper()} ", style=STATE_STYLE.get(self.state, "bold"))
        mic = Text(" 🔇 MIC MUTED  (/mic on) ", style="bold white on red") if self.mic_muted else Text("")
        left = Text.assemble(mic, st, "  ", (self.model, "bold"), "  ", (f"voice {self.voice}", "dim"))
        right = Text("  ".join(f"{k} {v}" for k, v in self.badges.items()), style="magenta")
        head.add_row(left, right)

        body = list(self.entries)
        if self.partial and self.state in ("user speaking", "interrupted"):
            p = Text()
            p.append("   you  ", style="bold cyan dim")
            p.append(self.partial, style="italic cyan dim")
            p.append(" ▍", style="cyan")
            body.append(p)
        if not body:
            body.append(Text("  say something…", style="dim italic"))

        bar = "█" * int(self.vad_p * 24)
        foot = Text.assemble(
            ("vad ", "dim"), (f"{bar:<24}", "cyan" if self.vad_p >= 0.5 else "grey50"), (f" {self.vad_p:4.2f}", "dim"),
            ("   echo ", "dim"), (f"{self.echo:4.2f}", "red" if self.echo > 0.45 else "dim"),
            ("   ", ""), (self.last_transition, "dim"),
            ("   ", ""), (self.status, "yellow"),
        )
        prompt = Text.assemble(("› ", "bold magenta"), (self.input_buf, "bold white"), ("▏", "magenta"),
                               ("   /help for commands", "dim") if not self.input_buf else ("", ""))
        return Group(Panel(head, padding=(0, 1)), Panel(Group(*body), title="conversation", padding=(0, 1)),
                     Panel(Group(foot, prompt), padding=(0, 1)))
