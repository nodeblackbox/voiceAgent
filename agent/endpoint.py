"""End-of-turn decision: Silero says "quiet", Smart Turn v3.2 says "finished".

    ep = Endpointer(min_silence_ms=250, max_silence_ms=1200, check_every_ms=200)
    ep.start_utterance()
    ...per 32 ms chunk...  decision = ep.update(utterance_audio, silence_ms, trailing_text)
                            -> None | "done" | "timeout"

Below `min_silence_ms` nothing happens. Between min and max, Smart Turn scores the last 8 s of audio every
`check_every_ms` of silence. At `max_silence_ms` the turn ends anyway (the fallback that keeps a
mis-scoring model from hanging the session). Without the model (`use_model=False`) it degrades to a plain
hangover of `max_silence_ms`.

Three confidence tiers, added after live use showed Smart Turn cutting people off mid-hedge ("um", "so",
"yeah yeah so..."):
  score <  threshold        -> not done
  threshold <= score < high  -> done, UNLESS the words spoken right before the silence end in a filler/
                                 continuation word ("um", "so", "and", "that's", ...) — then held, betting
                                 that a real speaker mid-hedge, not a finished thought, produced that score
  score >= high_threshold    -> done regardless of the trailing word (trust a confident model over a
                                 cheap lexical heuristic)
`trailing_text` is the most recent live partial transcript (a few hundred ms old, from the same ASR pass
already run for on-screen captions) — this costs nothing extra to check.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

# common hedges / conjunctions a real sentence doesn't end on — held unless Smart Turn is very confident
FILLER_ENDINGS = {
    "um", "umm", "uh", "uhh", "er", "ah", "so", "and", "but", "or", "the", "a", "is", "that", "that's",
    "like", "because", "which", "yeah", "well", "wait", "if", "i", "you", "we", "it's", "with", "for", "to",
}


def _last_word(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return words[-1] if words else ""


class Endpointer:
    def __init__(self, min_silence_ms: int = 250, max_silence_ms: int = 1200, check_every_ms: int = 200,
                 threshold: float = 0.7, high_threshold: float = 0.9, use_model: bool = True):
        self.min_ms, self.max_ms, self.every_ms = min_silence_ms, max_silence_ms, check_every_ms
        self.threshold, self.high_threshold = threshold, high_threshold
        self.model = None
        if use_model:
            from turn_test import SmartTurn

            self.model = SmartTurn()
            self.model(np.zeros(16000, dtype=np.float32))  # warm
        self.last_check_ms = -1e9
        self.last_p: float | None = None
        self.held_on_filler = 0  # how many times the filler-word guard has deferred a cut, for /status
        self.checks = 0
        self.cost_ms: list[float] = []

    def start_utterance(self) -> None:
        self.last_check_ms = -1e9
        self.last_p = None

    def update(self, audio: np.ndarray, silence_ms: float, trailing_text: str = "") -> str | None:
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
        if p < self.threshold:
            return None
        if p >= self.high_threshold:
            return "done"
        if _last_word(trailing_text) in FILLER_ENDINGS:
            self.held_on_filler += 1
            return None
        return "done"
