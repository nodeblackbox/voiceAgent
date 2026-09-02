"""A tiny non-blocking line editor for the terminal (Windows msvcrt, POSIX termios), so slash commands
can be typed while the Rich Live screen is running. Keys are read on a background thread; the current
buffer is shown in the UI footer; Enter hands the line to a callback.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque


class LineReader:
    def __init__(self, on_line, on_change=None):
        self.on_line, self.on_change = on_line, on_change
        self.buf = ""
        self.history: deque[str] = deque(maxlen=50)
        self._hist_i = 0
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True, name="keys")

    def start(self):
        self.t.start()

    def stop(self):
        self._stop = True

    def _emit(self):
        if self.on_change:
            self.on_change(self.buf)

    def _handle(self, ch: str):
        if ch in ("\r", "\n"):
            line, self.buf = self.buf, ""
            if line.strip():
                self.history.append(line)
            self._hist_i = len(self.history)
            self._emit()
            self.on_line(line)
        elif ch in ("\x08", "\x7f"):
            self.buf = self.buf[:-1]
            self._emit()
        elif ch == "\x15":  # ctrl-u
            self.buf = ""
            self._emit()
        elif ch == "\x03":  # ctrl-c
            raise KeyboardInterrupt
        elif ch == "UP":
            if self.history and self._hist_i > 0:
                self._hist_i -= 1
                self.buf = self.history[self._hist_i]
                self._emit()
        elif ch == "DOWN":
            if self._hist_i < len(self.history) - 1:
                self._hist_i += 1
                self.buf = self.history[self._hist_i]
            else:
                self._hist_i = len(self.history)
                self.buf = ""
            self._emit()
        elif ch.isprintable():
            self.buf += ch
            self._emit()

    def _run(self):
        if sys.platform == "win32":
            import msvcrt

            while not self._stop:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):  # arrow / function keys
                        code = msvcrt.getwch()
                        ch = {"H": "UP", "P": "DOWN"}.get(code, "")
                        if not ch:
                            continue
                    try:
                        self._handle(ch)
                    except KeyboardInterrupt:
                        self.on_line("/quit")
                        return
                else:
                    time.sleep(0.02)
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not self._stop:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        seq = sys.stdin.read(2)
                        ch = {"[A": "UP", "[B": "DOWN"}.get(seq, "")
                        if not ch:
                            continue
                    self._handle(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
