"""Voice agent: listen (Silero + Parakeet) -> think (LLM via LiteLLM, streaming) -> speak (Kokoro), with barge-in.

Turn timeline that gets measured:
    you stop talking ... [hangover] ... ASR ... LLM first sentence ... Kokoro ... first audio at the speaker
The headline number per turn is `speech-end -> first audio`.

Barge-in: while the agent is speaking, sustained speech on the mic (N consecutive VAD hits above a
higher threshold) interrupts: Kokoro's queue is flushed, the LLM stream is abandoned, and whatever was
already spoken is what goes into the conversation history.

Usage:
    python agent/voice_agent.py                                  # Groq default model, Yeti in/out
    python agent/voice_agent.py --model anthropic/claude-haiku-4-5
    python agent/voice_agent.py --say "what's a limit order"      # one turn from text, no mic (test)
    python agent/voice_agent.py --say "..." --no-play             # same, silent
    python agent/voice_agent.py --duration 60
Ctrl+C to stop. Turns are logged to results/agent_<timestamp>.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
for sub in ("bench", "tts", "llm"):
    sys.path.insert(0, str(ROOT / sub))
from common import RESULTS_DIR, assert_cuda_still_active, load_parakeet, pct, resolve_input_device, wasapi_settings  # noqa: E402
from kokoro_stream import KokoroTTS, NullSink, SentenceChunker, SpeakerSink, StreamingSpeech  # noqa: E402
from prompts import voice_system_prompt  # noqa: E402
from providers import DEFAULT_MODELS, StreamStats, stream_chat  # noqa: E402

con = Console()
SR = 16000
CHUNK = 512


def resolve_output_device(spec):
    import sounddevice as sd

    if spec is None:
        return None
    if str(spec).isdigit():
        return int(spec)
    apis = sd.query_hostapis()
    hits = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower()]
    hits.sort(key=lambda t: 0 if "WASAPI" in apis[t[1]["hostapi"]]["name"] else 1)
    if not hits:
        raise SystemExit(f"no output device matching '{spec}'")
    idx = hits[0][0]
    con.print(f"output device {idx}: {sd.query_devices(idx)['name']}")
    return idx


class Agent:
    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.system = voice_system_prompt(agent_name=args.agent_name, user_name=args.user_name)
        self.history: list[dict] = []
        self.log_path = RESULTS_DIR / f"agent_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.turn_n = 0
        self.metrics: list[dict] = []
        self.speaking = threading.Event()   # agent currently has audio queued/playing
        self._cancel = threading.Event()
        self._session: StreamingSpeech | None = None

        if not args.no_asr:
            con.print("[bold]Loading Parakeet[/] (hybrid) ...")
            self.asr, load_s, _ = load_parakeet("hybrid")
            self.asr.recognize(np.zeros(SR, dtype=np.float32))
            con.print(f"  loaded {load_s:.1f}s, providers after warm-up: {assert_cuda_still_active(self.asr, 'hybrid')}")
        con.print(f"[bold]Loading Kokoro[/] voice={args.voice} ...")
        self.tts = KokoroTTS(voice=args.voice)
        con.print(f"  loaded {self.tts.load_s:.1f}s warm-up {self.tts.warmup_s:.2f}s VRAM {self.tts.vram_mib():.0f} MiB")
        self.sink = NullSink() if args.no_play else SpeakerSink(device=resolve_output_device(args.out_device))
        con.print(f"[bold]LLM[/] {self.model}")
        # warm the HTTP connection so turn 1 doesn't pay TLS setup
        try:
            "".join(stream_chat(self.model, [{"role": "user", "content": "hi"}], system="Reply with one word.", max_tokens=5))
        except Exception as e:  # noqa: BLE001
            con.print(f"[red]LLM warm-up failed: {e}[/]")

    # ----------------------------------------------------------------- one turn
    def respond(self, user_text: str, t_speech_end: float | None = None, asr_ms: float | None = None) -> dict:
        self.turn_n += 1
        n = self.turn_n
        t_turn0 = time.perf_counter()
        self.history.append({"role": "user", "content": user_text})
        con.print(f"[bold green]#{n} you:[/] {user_text}")

        # reset any leftover session, start a fresh speaking session
        self._cancel.clear()
        sess = StreamingSpeech(self.tts, self.sink, speed=self.args.speed,
                               chunker=SentenceChunker(first_min_words=self.args.first_min_words))
        self._session = sess
        self.sink.finished = False
        self.sink.first_audio_wall = None
        self.speaking.set()

        st = StreamStats()
        spoken = ""
        interrupted = False
        try:
            for delta in stream_chat(self.model, self.history[-self.args.history:], system=self.system,
                                     max_tokens=self.args.max_tokens, stats=st):
                if self._cancel.is_set():
                    interrupted = True
                    break
                spoken += delta
                sess.feed(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            con.print(f"\n[red]LLM error: {type(e).__name__}: {str(e)[:200]}[/]")
            sess.feed("Sorry, I lost the connection to the model. Say that again in a moment.")
        print()
        if not interrupted:
            sess.end_of_text()
        m = sess.wait(timeout=120)
        self.speaking.clear()
        if m.interrupted_at_s is not None:
            interrupted = True
            # keep only what was actually said before the cut
            said = " ".join(s.text for s in m.sentences[: max(1, len(m.sentences))])
            spoken = said + " [interrupted]"
        self.history.append({"role": "assistant", "content": spoken.strip() or "[no reply]"})

        first_audio_from_turn = (self.sink.first_audio_wall - t_turn0) * 1e3 if self.sink.first_audio_wall else None
        rec = {
            "n": n, "user": user_text, "assistant": spoken.strip(), "model": self.model, "key": st.key_label,
            "asr_ms": None if asr_ms is None else round(asr_ms),
            "llm_ttft_ms": round(st.ttft_ms) if st.ttft_ms else None,
            "llm_first_sentence_ms": round(st.first_sentence_ms) if st.first_sentence_ms else None,
            "llm_total_ms": round(st.total_ms) if st.total_ms else None,
            "llm_tokens": st.completion_tokens, "llm_retries": st.retries,
            "tts_sentences": len(m.sentences),
            "tts_gen_ms_first": round(m.sentences[0].gen_s * 1e3) if m.sentences else None,
            "first_audio_from_turn_ms": round(first_audio_from_turn) if first_audio_from_turn else None,
            "speech_end_to_first_audio_ms": (round((self.sink.first_audio_wall - t_speech_end) * 1e3)
                                             if (t_speech_end and self.sink.first_audio_wall) else None),
            "underruns": m.underruns, "interrupted": interrupted, "wall": datetime.now().isoformat(),
        }
        self.metrics.append(rec)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        con.print(f"   [dim]asr {rec['asr_ms']} ms | llm first token {rec['llm_ttft_ms']} ms, first sentence "
                  f"{rec['llm_first_sentence_ms']} ms | kokoro {rec['tts_gen_ms_first']} ms | "
                  f"[bold]speech-end -> first audio {rec['speech_end_to_first_audio_ms']} ms[/] "
                  f"(from ASR done: {rec['first_audio_from_turn_ms']} ms){' | INTERRUPTED' if interrupted else ''}[/]")
        return rec

    def barge_in(self) -> None:
        if self._session is not None and self.speaking.is_set():
            self._cancel.set()
            self._session.interrupt()
            con.print("\n[yellow]-- barge-in: stopped speaking --[/]")

    # ----------------------------------------------------------------- mic loop
    def run_mic(self) -> None:
        import sounddevice as sd
        from silero_vad import load_silero_vad

        a = self.args
        vad = load_silero_vad()
        in_dev = resolve_input_device(a.in_device)
        audio_q: queue.Queue = queue.Queue()
        turn_q: queue.Queue = queue.Queue()

        def mic_cb(indata, frames, t, status):
            audio_q.put(indata[:, 0].copy())

        def worker():
            while True:
                item = turn_q.get()
                if item is None:
                    return
                audio, t_end = item
                t0 = time.perf_counter()
                text = self.asr.recognize(audio).strip()
                asr_ms = (time.perf_counter() - t0) * 1e3
                if not text:
                    con.print("[dim](nothing recognized)[/]")
                    continue
                self.respond(text, t_speech_end=t_end, asr_ms=asr_ms)

        threading.Thread(target=worker, daemon=True).start()
        preroll = collections.deque(maxlen=int(a.preroll * SR / CHUNK) + 1)
        utt: list[np.ndarray] = []
        speaking_user = False
        t_on = t_last = 0.0
        pending = np.zeros(0, dtype=np.float32)
        hot = 0  # consecutive VAD hits while the agent talks
        t0 = time.perf_counter()
        con.print(f"\n[bold]Listening[/] model={self.model} hangover={a.min_silence}ms preroll={a.preroll}s "
                  f"barge-in after {a.bargein_chunks} chunks > {a.bargein_threshold}. Ctrl+C to stop.\n")
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, device=in_dev,
                                extra_settings=wasapi_settings(in_dev), callback=mic_cb):
                while True:
                    pending = np.concatenate([pending, audio_q.get()])
                    if a.duration and time.perf_counter() - t0 > a.duration:
                        raise KeyboardInterrupt
                    while len(pending) >= CHUNK:
                        chunk, pending = pending[:CHUNK], pending[CHUNK:]
                        now = time.perf_counter()
                        p = vad(torch.from_numpy(chunk), SR).item()
                        agent_talking = self.speaking.is_set()
                        thr = a.bargein_threshold if agent_talking else a.threshold
                        if not speaking_user:
                            preroll.append(chunk)
                            if p >= thr:
                                hot += 1
                                if not agent_talking or hot >= a.bargein_chunks:
                                    if agent_talking:
                                        self.barge_in()
                                    speaking_user, t_on, t_last = True, now, now
                                    utt = list(preroll)
                                    sys.stdout.write(f"\r  [speech @ {now - t0:7.2f}s] ")
                                    sys.stdout.flush()
                            else:
                                hot = 0
                        else:
                            utt.append(chunk)
                            if p >= a.threshold:
                                t_last = now
                            if (now - t_last) * 1e3 >= a.min_silence or (now - t_on) >= a.max_utt:
                                speaking_user, hot = False, 0
                                audio = np.concatenate(utt)
                                if (t_last - t_on) * 1e3 >= a.min_speech:
                                    turn_q.put((audio, t_last))
                                    sys.stdout.write("end -> ASR -> LLM -> TTS\n")
                                else:
                                    sys.stdout.write("blip ignored\n")
                                preroll.clear()
                                vad.reset_states()
        except KeyboardInterrupt:
            pass
        finally:
            turn_q.put(None)
            self.sink.wait_drained(10)
            self.sink.close()
            self.summary()

    def summary(self) -> None:
        if not self.metrics:
            return
        con.print(f"\n[bold]Session[/] {len(self.metrics)} turns, model {self.model}")
        for k, label in [("speech_end_to_first_audio_ms", "speech-end -> first audio"),
                         ("first_audio_from_turn_ms", "ASR done -> first audio"),
                         ("llm_first_sentence_ms", "LLM first sentence"), ("asr_ms", "ASR")]:
            v = [m[k] for m in self.metrics if m.get(k) is not None]
            if v:
                con.print(f"  {label:26} p50 {pct(v, 50):5.0f} ms   p95 {pct(v, 95):5.0f} ms")
        con.print(f"  interrupted turns: {sum(1 for m in self.metrics if m['interrupted'])}")
        con.print(f"  log -> {self.log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODELS["groq"])
    ap.add_argument("--agent-name", default="Yeti")
    ap.add_argument("--user-name", default="Nasan")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--history", type=int, default=20, help="messages of history sent to the model")
    ap.add_argument("--first-min-words", type=int, default=6)
    ap.add_argument("--in-device", default="Yeti")
    ap.add_argument("--out-device", default="Yeti")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bargein-threshold", type=float, default=0.8)
    ap.add_argument("--bargein-chunks", type=int, default=4, help="consecutive 32 ms VAD hits needed to interrupt")
    ap.add_argument("--min-silence", type=int, default=600)
    ap.add_argument("--min-speech", type=int, default=250)
    ap.add_argument("--preroll", type=float, default=1.0)
    ap.add_argument("--max-utt", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--say", nargs="*", default=None, help="text turns instead of the mic (each arg is one turn)")
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--no-asr", action="store_true", help="skip loading Parakeet (only valid with --say)")
    args = ap.parse_args()
    if args.say is None:
        args.no_asr = False

    agent = Agent(args)
    if args.say is not None:
        for text in args.say:
            agent.respond(text, t_speech_end=time.perf_counter())
        agent.sink.wait_drained(60)
        agent.sink.close()
        agent.summary()
    else:
        agent.run_mic()


if __name__ == "__main__":
    main()
