"""The interrupt state machine. Every transition is explicit and logged, so barge-in is never a race.

States
  LISTENING       mic open, nobody talking
  USER_SPEAKING   VAD is on; partials are being transcribed for the UI
  THINKING        utterance closed, ASR done, LLM streaming, no audio out yet
  SPEAKING        Kokoro audio is playing (LLM may still be streaming behind it)
  INTERRUPTED     user barged in; agent is being cut; transient, returns to USER_SPEAKING

Events
  vad_on, vad_off (after hangover), asr_done, first_audio, playback_done, barge_in, error

A barge-in is only accepted from THINKING or SPEAKING and only when `barge_in_ok()` agrees (sustained
speech, not echo). What the agent had already said is kept; the unspoken remainder is dropped but
remembered so the model can pick the thread back up if asked.
"""
from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field


class S(enum.Enum):
    LISTENING = "listening"
    USER_SPEAKING = "user speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class Turn:
    n: int
    user_text: str = ""
    user_text_for_model: str = ""
    partial: str = ""                       # live transcript while speaking
    assistant_text: str = ""                # everything the model streamed
    spoken_text: str = ""                   # what actually came out of the speaker
    unspoken_text: str = ""                 # streamed but cut by barge-in
    tool_calls: list[str] = field(default_factory=list)
    interrupted: bool = False
    t_speech_end: float | None = None
    asr_ms: float | None = None
    llm_first_token_ms: float | None = None
    llm_first_sentence_ms: float | None = None
    tts_first_ms: float | None = None
    speech_end_to_audio_ms: float | None = None
    started: float = field(default_factory=time.perf_counter)


class StateMachine:
    def __init__(self, on_change=None):
        self.state = S.LISTENING
        self.lock = threading.Lock()
        self.on_change = on_change
        self.log: list[tuple[float, str, str, str]] = []
        self.t0 = time.perf_counter()

    def _set(self, new: S, event: str) -> None:
        old = self.state
        self.state = new
        self.log.append((time.perf_counter() - self.t0, event, old.value, new.value))
        if self.on_change:
            self.on_change(old, new, event)

    # transitions ------------------------------------------------------------------------------
    def vad_on(self) -> bool:
        with self.lock:
            if self.state == S.LISTENING:
                self._set(S.USER_SPEAKING, "vad_on")
                return True
            return False

    def barge_in(self) -> bool:
        with self.lock:
            if self.state in (S.THINKING, S.SPEAKING):
                self._set(S.INTERRUPTED, "barge_in")
                return True
            return False

    def resume_user(self) -> None:
        """INTERRUPTED -> USER_SPEAKING once the cut is done."""
        with self.lock:
            if self.state == S.INTERRUPTED:
                self._set(S.USER_SPEAKING, "user_continues")

    def force_listening(self, event: str = "forced") -> None:
        """Abandon whatever the mic side was doing (e.g. mic muted mid-utterance) and go idle.
        Never used on THINKING/SPEAKING — muting the mic must not cut the agent's own reply."""
        with self.lock:
            if self.state in (S.USER_SPEAKING, S.INTERRUPTED):
                self._set(S.LISTENING, event)

    def abort(self) -> None:
        """INTERRUPTED -> LISTENING (a /stop command, not a user utterance)."""
        with self.lock:
            if self.state == S.INTERRUPTED:
                self._set(S.LISTENING, "stopped")

    def vad_off(self) -> bool:
        with self.lock:
            if self.state == S.USER_SPEAKING:
                self._set(S.THINKING, "vad_off")
                return True
            return False

    def blip(self) -> None:
        with self.lock:
            if self.state == S.USER_SPEAKING:
                self._set(S.LISTENING, "blip")

    def first_audio(self) -> None:
        with self.lock:
            if self.state == S.THINKING:
                self._set(S.SPEAKING, "first_audio")

    def done(self) -> None:
        with self.lock:
            if self.state in (S.THINKING, S.SPEAKING):
                self._set(S.LISTENING, "playback_done")

    def error(self) -> None:
        with self.lock:
            self._set(S.LISTENING, "error")

    @property
    def agent_busy(self) -> bool:
        return self.state in (S.THINKING, S.SPEAKING)
