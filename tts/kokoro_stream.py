"""Kokoro-82M streaming TTS for a voice agent.

Design
------
Kokoro synthesises one phoneme sequence at a time (whole segment in, whole waveform out), so the
"streaming" that matters for an agent is *pipelining at sentence granularity*:

    LLM tokens -> SentenceChunker -> [gen thread: Kokoro on GPU] -> queue -> [audio callback: speaker]

Sentence N+1 is synthesised while sentence N is playing. Time-to-first-audio is therefore the cost of
the first sentence only. Barge-in is `interrupt()`: flush the queue, stop generating, playback stops
at the next audio callback (<= blocksize, ~21 ms at 24 kHz / 512).

Everything is float32 mono at 24 kHz. No effects, no cache, no Flask: one model, kept resident.
"""
from __future__ import annotations

import collections
import queue
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

SR = 24000
REPO = "hexgrad/Kokoro-82M"


# --------------------------------------------------------------------------- text -> sentences
class SentenceChunker:
    """Feed arbitrary text fragments (LLM tokens); get back complete sentences as they close.

    Rules: a sentence ends at . ! ? … (optionally followed by quotes/brackets) + whitespace, or at a
    newline. If a sentence runs past `max_chars` we cut at the last comma/semicolon/space so Kokoro
    never gets a wall of text. `first_min_words`: release the *first* chunk early at a clause break
    (comma/colon/dash) once it has that many words, to shave time-to-first-audio.
    """

    END = re.compile(r'([.!?…]+["\')\]]*)(\s+|$)')
    CLAUSE = re.compile(r'([,;:—–-])\s+')

    def __init__(self, max_chars: int = 220, first_min_words: int = 6):
        self.buf = ""
        self.max_chars = max_chars
        self.first_min_words = first_min_words
        self.emitted = 0

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        out: list[str] = []
        while True:
            m = self.END.search(self.buf)
            if m and (m.end() < len(self.buf) or m.group(2)):  # boundary confirmed by following whitespace
                out.append(self.buf[: m.end()].strip())
                self.buf = self.buf[m.end():]
                continue
            nl = self.buf.find("\n")
            if nl >= 0:
                s = self.buf[:nl].strip()
                self.buf = self.buf[nl + 1:]
                if s:
                    out.append(s)
                continue
            if self.emitted == 0 and not out and self.first_min_words:
                cm = self.CLAUSE.search(self.buf)
                if cm and len(self.buf[: cm.start()].split()) >= self.first_min_words:
                    out.append(self.buf[: cm.end()].strip())
                    self.buf = self.buf[cm.end():]
                    continue
            if len(self.buf) > self.max_chars:
                cut = max(self.buf.rfind(", ", 0, self.max_chars), self.buf.rfind(" ", 0, self.max_chars))
                if cut <= 0:
                    cut = self.max_chars
                out.append(self.buf[:cut].strip())
                self.buf = self.buf[cut:].lstrip()
                continue
            break
        self.emitted += len(out)
        return [s for s in out if s]

    def flush(self) -> list[str]:
        s, self.buf = self.buf.strip(), ""
        if s:
            self.emitted += 1
            return [s]
        return []


# --------------------------------------------------------------------------- model
class KokoroTTS:
    """One resident Kokoro model + pipeline. `synth()` returns float32 24 kHz audio for a text."""

    def __init__(self, voice: str = "af_heart", device: str = "cuda", lang: str | None = None):
        from kokoro import KModel, KPipeline

        if device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("FATAL: torch.cuda.is_available() is False; refusing to fall back to CPU silently")
        self.device = device
        self.voice = voice
        lang = lang or voice[0]  # 'a' = American English, 'b' = British
        t0 = time.perf_counter()
        self.model = KModel(repo_id=REPO).to(device).eval()
        self.pipeline = KPipeline(lang_code=lang, model=self.model, repo_id=REPO)
        self.pipeline.load_voice(voice)
        self.load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.synth("Warm up.")
        if device == "cuda":
            torch.cuda.synchronize()
        self.warmup_s = time.perf_counter() - t0

    def set_voice(self, voice: str) -> str:
        """Switch voice without reloading the model (voices are small tensors)."""
        lang = voice[0]
        if lang != self.voice[0]:
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code=lang, model=self.model, repo_id=REPO)
        self.pipeline.load_voice(voice)
        self.voice = voice
        self.synth("Okay.")
        return voice

    def vram_mib(self) -> float | None:
        if self.device != "cuda":
            return None
        return torch.cuda.memory_allocated() / 2**20

    @torch.inference_mode()
    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        parts = []
        for r in self.pipeline(text, voice=self.voice, speed=speed, split_pattern=r"\n+"):
            if r.audio is not None:
                parts.append(r.audio.detach().float().cpu().numpy())
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- sinks
class SpeakerSink:
    """sounddevice OutputStream fed from a queue of float32 chunks. Counts underruns (silence gaps)."""

    def __init__(self, device=None, blocksize: int = 512, on_play=None):
        import sounddevice as sd

        self.on_play = on_play  # callback(float32 24 kHz block actually sent to the DAC), for the echo gate
        self.gain = 1.0         # 1.0 normal; ~0.25 while "ducked" for a possible barge-in
        self.q: collections.deque[np.ndarray] = collections.deque()
        self.lock = threading.Lock()
        self.pending = np.zeros(0, dtype=np.float32)
        self.underruns = 0
        self.played_samples = 0
        self.first_audio_wall: float | None = None
        self.started = False
        extra = None
        try:
            dev = sd.query_devices(device if device is not None else sd.default.device[1])
            if "WASAPI" in sd.query_hostapis(dev["hostapi"])["name"]:
                extra = sd.WasapiSettings(auto_convert=True)  # WASAPI shared mode won't accept 24 kHz otherwise
        except Exception:
            pass
        self.stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=blocksize,
                                      device=device, extra_settings=extra, callback=self._cb)
        self.stream.start()

    def _cb(self, outdata, frames, t, status):
        need = frames
        out = np.zeros(frames, dtype=np.float32)
        pos = 0
        with self.lock:
            while need > 0:
                if len(self.pending) == 0:
                    if not self.q:
                        break
                    self.pending = self.q.popleft()
                n = min(need, len(self.pending))
                out[pos:pos + n] = self.pending[:n]
                self.pending = self.pending[n:]
                pos += n
                need -= n
        if pos > 0 and self.first_audio_wall is None:
            self.first_audio_wall = time.perf_counter()
        if pos > 0:
            self.started = True
        elif self.started and not self.finished:
            self.underruns += 1  # a callback with nothing to play while the generator still owes audio
        self.played_samples += pos
        if self.gain != 1.0:
            out *= self.gain
        if pos > 0 and self.on_play is not None:
            try:
                self.on_play(out[:pos])
            except Exception:  # noqa: BLE001  never let the gate break the audio callback
                pass
        outdata[:, 0] = out

    drained_flag = False
    finished = False

    def push(self, audio: np.ndarray) -> None:
        with self.lock:
            self.q.append(audio)

    def queued_samples(self) -> int:
        return sum(len(a) for a in self.q) + len(self.pending)

    def flush(self) -> None:
        with self.lock:
            self.q.clear()
            self.pending = np.zeros(0, dtype=np.float32)

    def wait_drained(self, timeout: float = 60) -> None:
        t0 = time.perf_counter()
        while self.queued_samples() > 0 and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)
        self.drained_flag = True
        time.sleep(0.05)

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class NullSink:
    """Same interface, no sound card: consumes audio at real-time pace so gaps still get measured."""

    gain = 1.0

    def __init__(self, on_play=None, **_):
        self.on_play = on_play
        self.q: collections.deque[np.ndarray] = collections.deque()
        self.lock = threading.Lock()
        self.underruns = 0
        self.first_audio_wall = None
        self.played_samples = 0
        self._stop = False
        self.started = False
        self.drained_flag = False
        self.finished = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        block = 512
        next_t = time.perf_counter()
        while not self._stop:
            with self.lock:
                have = sum(len(a) for a in self.q)
                if have > 0:
                    need = min(block, have)
                    while need > 0:
                        a = self.q[0]
                        n = min(need, len(a))
                        self.q[0] = a[n:]
                        if len(self.q[0]) == 0:
                            self.q.popleft()
                        need -= n
                    got = True
                else:
                    got = False
            if got:
                if self.first_audio_wall is None:
                    self.first_audio_wall = time.perf_counter()
                self.started = True
                self.played_samples += block
            elif self.started and not self.finished:
                self.underruns += 1  # starved while the generator still owes audio
            next_t += block / SR
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def push(self, audio):
        with self.lock:
            self.q.append(audio.copy())

    def queued_samples(self):
        with self.lock:
            return sum(len(a) for a in self.q)

    def flush(self):
        with self.lock:
            self.q.clear()

    def wait_drained(self, timeout=60):
        t0 = time.perf_counter()
        while self.queued_samples() > 0 and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)
        self.drained_flag = True
        time.sleep(0.05)

    def close(self):
        self._stop = True


# --------------------------------------------------------------------------- session
@dataclass
class SentenceMetric:
    idx: int
    text: str
    chars: int
    t_ready: float        # when the sentence was complete (relative to session start)
    gen_s: float          # Kokoro time
    audio_s: float
    rtf: float            # gen_s / audio_s (lower is better; < 1 keeps up)


@dataclass
class SessionMetrics:
    t_first_text: float | None = None
    t_first_audio: float | None = None
    t_last_text: float | None = None
    t_done: float | None = None
    underruns: int = 0
    interrupted_at_s: float | None = None
    sentences: list[SentenceMetric] = field(default_factory=list)

    @property
    def ttfa_ms(self) -> float | None:
        if self.t_first_text is None or self.t_first_audio is None:
            return None
        return (self.t_first_audio - self.t_first_text) * 1e3


class StreamingSpeech:
    """Consume a text stream, speak it, and measure. One instance per utterance/turn."""

    def __init__(self, tts: KokoroTTS, sink, speed: float = 1.0, chunker: SentenceChunker | None = None):
        self.tts, self.sink, self.speed = tts, sink, speed
        self.chunker = chunker or SentenceChunker()
        self.m = SessionMetrics()
        self._q: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._gen = threading.Thread(target=self._gen_loop, daemon=True)
        self._gen.start()

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def feed(self, piece: str) -> None:
        if self.m.t_first_text is None:
            self.m.t_first_text = self._now()
        self.m.t_last_text = self._now()
        for s in self.chunker.feed(piece):
            self._q.put(s)

    def flush(self) -> None:
        """Speak whatever is buffered now (e.g. 'Let me check.' right before a tool call)."""
        for s in self.chunker.flush():
            self._q.put(s)

    def end_of_text(self) -> None:
        self.flush()
        self._q.put(None)

    def interrupt(self) -> None:
        """Barge-in: stop generating and drop everything queued at the speaker."""
        self._stop.set()
        self.sink.flush()
        self.m.interrupted_at_s = self._now()
        self._q.put(None)

    def _gen_loop(self) -> None:
        idx = 0
        while not self._stop.is_set():
            s = self._q.get()
            if s is None:
                break
            t_ready = self._now()
            t0 = time.perf_counter()
            if hasattr(self.tts, "synth_stream"):
                # engine streams its own sub-sentence chunks (e.g. NeuTTS): push each as it arrives
                # instead of waiting for the whole sentence — finer-grained than Kokoro's per-sentence
                # pipelining. `gen` here is time-to-first-chunk, the number that actually matters for
                # first-audio latency; `dur` is the full sentence once every chunk has arrived.
                first_gen, total_samples = None, 0
                for chunk in self.tts.synth_stream(s, self.speed):
                    if self._stop.is_set():
                        break
                    if first_gen is None:
                        first_gen = time.perf_counter() - t0
                    self.sink.push(chunk)
                    total_samples += len(chunk)
                if self._stop.is_set():
                    break
                gen = first_gen if first_gen is not None else (time.perf_counter() - t0)
                dur = total_samples / SR
            else:
                audio = self.tts.synth(s, self.speed)
                gen = time.perf_counter() - t0
                if self._stop.is_set():
                    break
                self.sink.push(audio)
                dur = len(audio) / SR
            self.m.sentences.append(SentenceMetric(idx, s, len(s), t_ready, gen, dur, gen / dur if dur else 0.0))
            idx += 1
        self.sink.finished = True

    def wait(self, timeout: float = 120) -> SessionMetrics:
        self._gen.join(timeout)
        if not self._stop.is_set():
            self.sink.wait_drained(timeout)
        if self.sink.first_audio_wall is not None:
            self.m.t_first_audio = self.sink.first_audio_wall - self._t0
        self.m.underruns = self.sink.underruns
        self.m.t_done = self._now()
        return self.m
