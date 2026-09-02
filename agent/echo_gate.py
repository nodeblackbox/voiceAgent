"""Echo gate: stops the agent from barging in on itself.

When the agent's own voice leaks from the speakers into the mic, Silero happily calls it speech. We
keep the last second of what the speaker actually played (as a 16 kHz reference) and, for each mic
chunk, check how well it correlates with the reference at any plausible acoustic delay. High
correlation = the mic is hearing the agent, not the user, and the barge-in logic ignores the chunk.

This is echo *detection* for gating, not echo *cancellation*: it never edits the audio that goes to
Parakeet. With headphones the gate almost never fires; with open speakers it is what keeps the agent
from interrupting itself mid-sentence. Cost: one normalised cross-correlation per 32 ms chunk.
"""
from __future__ import annotations

import threading

import numpy as np

SR = 16000


class EchoGate:
    def __init__(self, ref_seconds: float = 1.0, max_delay_ms: int = 350, corr_threshold: float = 0.45,
                 level_ratio: float = 3.0):
        self.ref_len = int(ref_seconds * SR)
        self.max_delay = int(max_delay_ms / 1000 * SR)
        self.corr_threshold = corr_threshold
        self.level_ratio = level_ratio  # mic must be this much louder than expected echo to count as user
        self.ref = np.zeros(self.ref_len, dtype=np.float32)
        self.lock = threading.Lock()
        self.active = False       # True while the speaker is playing agent audio
        self.last_corr = 0.0
        self.last_delay_ms = 0.0
        self.suppressed = 0

    # --- reference side (called by the speaker sink with what it just played, at 24 kHz) ------------
    def push_played(self, audio24k: np.ndarray) -> None:
        if len(audio24k) == 0:
            return
        n = int(len(audio24k) * SR / 24000)
        idx = np.linspace(0, len(audio24k) - 1, n)
        a16 = np.interp(idx, np.arange(len(audio24k)), audio24k).astype(np.float32)
        with self.lock:
            self.ref = np.concatenate([self.ref, a16])[-self.ref_len:]

    def set_active(self, on: bool) -> None:
        self.active = on
        if not on:
            with self.lock:
                self.ref[:] = 0.0

    # --- mic side --------------------------------------------------------------------------------
    def is_echo(self, chunk: np.ndarray) -> bool:
        """True if this mic chunk looks like the speaker's own output."""
        if not self.active:
            return False
        with self.lock:
            ref = self.ref.copy()
        if float(np.max(np.abs(ref))) < 1e-4:
            return False
        c = chunk - chunk.mean()
        cn = float(np.linalg.norm(c))
        if cn < 1e-6:
            return False
        # search the most recent (max_delay + chunk) samples of the reference
        win = ref[-(self.max_delay + len(c)):]
        corr = np.correlate(win, c, mode="valid")  # one value per candidate delay
        norms = np.sqrt(np.convolve(win.astype(np.float64) ** 2, np.ones(len(c)), mode="valid")) + 1e-9
        ncc = corr / (norms * cn)
        k = int(np.argmax(np.abs(ncc)))
        best = float(abs(ncc[k]))
        self.last_corr = best
        self.last_delay_ms = (len(ncc) - 1 - k) / SR * 1e3
        # level check: if the mic is far louder than the reference segment it matched, a person is talking over it
        ref_rms = float(np.sqrt(np.mean(win[k:k + len(c)] ** 2)) + 1e-9)
        mic_rms = float(np.sqrt(np.mean(chunk ** 2)))
        if best >= self.corr_threshold and mic_rms < ref_rms * self.level_ratio:
            self.suppressed += 1
            return True
        return False
