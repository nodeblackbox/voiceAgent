"""End-of-turn decision: Silero says "quiet", Smart Turn v3.2 says "finished".

    ep = Endpointer(min_silence_ms=250, max_silence_ms=1200, check_every_ms=200)
    ep.start_utterance()
    ...per 32 ms chunk...  decision = ep.update(utterance_audio, silence_ms)  -> None | "done" | "timeout"

Below `min_silence_ms` nothing happens. Between min and max, Smart Turn scores the last 8 s of audio every
`check_every_ms` of silence; a score above `threshold` ends the turn. At `max_silence_ms` the turn ends
anyway (the fallback that keeps a mis-scoring model from hanging the session). Without the model
(`use_model=False`) it degrades to a plain hangover of `max_silence_ms`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))


class Endpointer:
    def __init__(self, min_silence_ms: int = 250, max_silence_ms: int = 1200, check_every_ms: int = 200,
                 threshold: float = 0.5, use_model: bool = True):
        self.min_ms, self.max_ms, self.every_ms, self.threshold = min_silence_ms, max_silence_ms, check_every_ms, threshold
        self.model = None
        if use_model:
            from turn_test import SmartTurn

            self.model = SmartTurn()
            self.model(np.zeros(16000, dtype=np.float32))  # warm
        self.last_check_ms = -1e9
        self.last_p: float | None = None
        self.checks = 0
        self.cost_ms: list[float] = []

    def start_utterance(self) -> None:
        self.last_check_ms = -1e9
        self.last_p = None

    def update(self, audio: np.ndarray, silence_ms: float) -> str | None:
        if silence_ms < self.min_ms:
            return None
        if silence_ms >= self.max_ms:
            return "timeout"
        if self.model is None:
            return None
        if silence_ms - self.last_check_ms < self.every_ms:
            return None
        self.last_check_ms = silence_ms
        t0 = time.perf_counter()
        p = self.model(audio)
        self.cost_ms.append((time.perf_counter() - t0) * 1e3)
        self.checks += 1
        self.last_p = p
        return "done" if p >= self.threshold else None
