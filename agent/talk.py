"""talk.py: the conversational agent with barge-in.

    mic ─ Silero ─┬─ EchoGate ─ StateMachine ─ barge-in
                  ├─ live partials (Parakeet on the growing utterance, for the UI)
                  └─ utterance ─ Parakeet ─ Brain (LangGraph + tools, streaming) ─ Kokoro ─ speaker

Usage
    python agent/talk.py                                       # Claude Haiku 4.5, Yeti in/out
    python agent/talk.py --model groq:openai/gpt-oss-20b
    python agent/talk.py --say "what time is it" "remember that I like the af_bella voice"
    python agent/talk.py --sim audio/handy-1787878656.wav --sim-bargein audio/handy-1787881187.wav
        (simulated mic: first wav is an utterance; second is injected 1.2 s after the agent starts talking)
Ctrl+C to stop. Every turn goes to results/talk_<timestamp>.jsonl.
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

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; the UI draws unicode
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parent.parent
for sub in ("bench", "tts", "llm", "agent"):
    sys.path.insert(0, str(ROOT / sub))
from brain import Brain  # noqa: E402
from commands import CommandRouter  # noqa: E402
from keys import LineReader  # noqa: E402
from common import RESULTS_DIR, assert_cuda_still_active, load_parakeet, load_wav_16k, pct, resolve_input_device, wasapi_settings  # noqa: E402
from echo_gate import EchoGate  # noqa: E402
from kokoro_stream import SR as TTS_SR, KokoroTTS, NullSink, SentenceChunker, SpeakerSink, StreamingSpeech  # noqa: E402
from prompts import voice_system_prompt  # noqa: E402
from state import S, StateMachine, Turn  # noqa: E402
from ui import TerminalUI  # noqa: E402

SR = 16000
CHUNK = 512

SEARCH_PROMPT_ADDON = """
Web
- When web tools are available (web_search, read_page, research), use them for anything current, anything after your training data, prices, versions, news, and facts you are not sure of. Say a short lead-in first ("Let me look that up.").
- Speak the source by name ("according to the NVIDIA page"), never read out a URL.
- Be economical: one web_search plus at most one read_page usually answers a question. The moment you have the answer, say it; do not re-search to double-check unless the user asked for certainty. Use research(question) when you genuinely need several pages.
- If a page is blocked or unreadable, say so and answer from what you have rather than trying the same page again.
"""

INTERRUPT_PROMPT_ADDON = """
Interruptions
- If your previous message ends with a note in square brackets saying you were interrupted, the user heard only the text before the note. Do not repeat what they heard. If they ask you to go on or finish, continue from where you were cut, briefly. If they changed the subject, follow them and forget the rest.
- When you use a tool, say a short natural phrase first ("Let me check.") so the pause is not silent.
"""


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
    return hits[0][0]


class Talk:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.ui = TerminalUI(model=a.model, voice=a.voice, plain=a.plain)
        self.sm = StateMachine(on_change=lambda old, new, ev: self.ui.set_state(new.value, f"{old.value} → {new.value} ({ev})"))
        self.echo = EchoGate(corr_threshold=a.echo_corr)
        self.log_path = RESULTS_DIR / f"talk_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.turns: list[Turn] = []
        self.turn_n = 0
        self.cancel = threading.Event()
        self.session: StreamingSpeech | None = None
        self.session_play0 = 0
        self.asr_lock = threading.Lock()
        self.turn_q: queue.Queue = queue.Queue()
        self.muted = False
        self.quit = threading.Event()
        self.router = self._build_commands()

    # ---------------------------------------------------------------- loading
    def load(self):
        a = self.a
        self.ui.set_status("loading Parakeet…")
        if a.say is None:
            self.asr, _, _ = load_parakeet("hybrid")
            self.asr.recognize(np.zeros(SR, dtype=np.float32))
            assert_cuda_still_active(self.asr, "hybrid")
        self.ui.set_status("loading Kokoro…")
        self.tts = KokoroTTS(voice=a.voice)
        self.sink = NullSink(on_play=self.echo.push_played) if a.no_play else \
            SpeakerSink(device=resolve_output_device(a.out_device), on_play=self.echo.push_played)
        self.ui.set_status("connecting to model…")
        system = voice_system_prompt(agent_name=a.agent_name, user_name=a.user_name) + SEARCH_PROMPT_ADDON + INTERRUPT_PROMPT_ADDON
        self.brain = Brain(a.model, system_prompt=system, max_tokens=a.max_tokens, search=False)
        self.brain.on_notice = lambda msg: self.ui.note(msg, "yellow")
        self.brain.warm()
        if a.search:
            self.ui.note(self._search_on(), "dim")
        for msg in self.brain.mcp_autoconnect():
            self.ui.note(f"mcp: {msg}", "yellow dim")
        self.ui.set_status("")
        self.ui.note(f"ready · Parakeet hybrid · Kokoro {a.voice} ({self.tts.vram_mib():.0f} MiB) · {a.model} · "
                     f"tools: {', '.join(n for n, _ in self.brain.tool_sources())} · type /help", "green dim")

    def _search_on(self) -> str:
        sys.path.insert(0, str(ROOT / "search"))
        from run_searxng import ensure_running

        self.ui.set_status("starting SearXNG…")
        up, msg = ensure_running()
        self.ui.set_status("")
        if not up:
            return f"search NOT enabled: {msg}"
        self.brain.set_search(True)
        return f"search on ({msg})"

    # ---------------------------------------------------------------- slash commands
    def _build_commands(self) -> CommandRouter:
        r = CommandRouter()

        @r.add("help", "help", "list commands")
        def _help(args):
            return r.help_text()

        @r.add("model", "model <provider:model>", "switch the model, e.g. groq:openai/gpt-oss-20b")
        def _model(args):
            if not args:
                return f"model is {self.brain.model_spec}"
            self.ui.set_status("switching model…")
            try:
                spec = self.brain.set_model(args[0])
            finally:
                self.ui.set_status("")
            self.ui.model = spec
            return f"model → {spec}"

        @r.add("voice", "voice <name>", "switch the Kokoro voice, e.g. af_bella, am_michael, bf_emma")
        def _voice(args):
            if not args:
                return f"voice is {self.tts.voice}"
            self.tts.set_voice(args[0])
            self.ui.voice = args[0]
            return f"voice → {args[0]}"

        @r.add("tools", "tools", "list the tools the model has right now")
        def _tools(args):
            return "\n".join(f"  {n:16} {src}" for n, src in self.brain.tool_sources()) or "  (none)"

        @r.add("search", "search on|off", "give or take away the web tools (starts SearXNG if needed)")
        def _search(args):
            if not args:
                return f"search is {'on' if self.brain.search_on else 'off'}"
            if args[0] == "on":
                return self._search_on()
            self.brain.set_search(False)
            return "search off"

        @r.add("mcp", "mcp [on|off <name> | reload]", "list MCP servers from mcp.json, or switch one")
        def _mcp(args):
            if not args:
                rows = self.brain.mcp_status()
                return "\n".join(f"  {'●' if x['on'] else '○'} {x['name']:12} {x['transport']:8} "
                                  f"{', '.join(x['tools']) if x['on'] else x['desc']}" for x in rows) or "  (no servers in mcp.json)"
            if args[0] == "reload":
                from brain import load_mcp_config
                self.brain.mcp_cfg = load_mcp_config()
                return f"reloaded mcp.json: {', '.join(self.brain.mcp_cfg)}"
            if len(args) < 2:
                return "usage: /mcp on <name> | off <name>"
            self.ui.set_status(f"mcp {args[0]} {args[1]}…")
            try:
                return self.brain.mcp_on(args[1]) if args[0] == "on" else self.brain.mcp_off(args[1])
            finally:
                self.ui.set_status("")

        @r.add("say", "say <text>", "send a typed turn instead of speaking")
        def _say(args):
            text = " ".join(args).strip()
            if not text:
                return "say what?"
            if self.sm.agent_busy:
                self.barge_in()
                self.sm.abort()
            self.sm.vad_on()
            self.sm.vad_off()
            self.turn_q.put((None, time.perf_counter(), text))
            return None

        @r.add("stop", "stop", "interrupt the current reply")
        def _stop(args):
            if self.sm.agent_busy:
                self.barge_in()
                self.sm.abort()
                return "stopped"
            return "nothing playing"

        @r.add("mute", "mute", "stop speaking replies (text still streams)")
        def _mute(args):
            self.muted = True
            return "muted"

        @r.add("unmute", "unmute", "speak replies again")
        def _unmute(args):
            self.muted = False
            return "unmuted"

        @r.add("history", "history [n]", "show the last n exchanges")
        def _history(args):
            n = int(args[0]) if args else 4
            return "\n".join(f"  {who:5} {txt[:140]}" for who, txt in self.brain.transcript(n)) or "  (empty)"

        @r.add("clear", "clear", "forget the conversation")
        def _clear(args):
            self.brain.reset()
            return "history cleared"

        @r.add("status", "status", "state, model, last-turn timings")
        def _status(args):
            last = self.turns[-1] if self.turns else None
            lt = (f"last turn: asr {last.asr_ms and round(last.asr_ms)} ms, llm first sentence "
                  f"{last.llm_first_sentence_ms and round(last.llm_first_sentence_ms)} ms, end→audio "
                  f"{last.speech_end_to_audio_ms and round(last.speech_end_to_audio_ms)} ms") if last else "no turns yet"
            return (f"state {self.sm.state.value} · model {self.brain.model_spec} · voice {self.tts.voice} · "
                    f"search {'on' if self.brain.search_on else 'off'} · mcp {', '.join(self.brain.mcp_tools) or 'none'} · "
                    f"{'muted' if self.muted else 'sound on'} · echo gate suppressed {self.echo.suppressed}\n  {lt}")

        @r.add("quit", "quit", "exit")
        def _quit(args):
            self.quit.set()
            return "bye"

        return r

    def on_command_line(self, line: str) -> None:
        out = self.router.dispatch(line)
        if out:
            for ln in str(out).split("\n"):
                self.ui.note(ln, "bold" if ln.startswith("/") else "white")

    # ---------------------------------------------------------------- one turn (runs in the worker thread)
    def respond(self, turn: Turn):
        a = self.a
        self.cancel.clear()
        t_turn0 = time.perf_counter()
        self.ui.assistant_start(turn.n)
        sink = NullSink() if self.muted else self.sink
        sess = StreamingSpeech(self.tts, sink, speed=a.speed, chunker=SentenceChunker(first_min_words=a.first_min_words))
        self.session = sess
        self.turn_sink = sink
        self.session_play0 = sink.played_samples
        sink.finished = False
        sink.first_audio_wall = None
        self.echo.set_active(not self.muted)

        def on_text(d):
            sess.feed(d)
            self.ui.assistant_text(d)
            if self.turn_sink.first_audio_wall and self.sm.state == S.THINKING:
                self.sm.first_audio()

        def on_tool_call(name):
            sess.flush()  # say the lead-in phrase now, not after the tool returns
            self.ui.tool(name)

        def on_tool_result(name, res):
            self.ui.tool(name, res)

        # watch for first audio while the LLM is still streaming (audio can start before the next delta)
        def watch_audio():
            while self.session is sess and not sess.m.t_done:
                if self.turn_sink.first_audio_wall and self.sm.state == S.THINKING:
                    self.sm.first_audio()
                    break
                time.sleep(0.01)
        threading.Thread(target=watch_audio, daemon=True).start()

        r = self.brain.stream_turn(turn.user_text, on_text=on_text, on_tool_call=on_tool_call,
                                   on_tool_result=on_tool_result, cancel=self.cancel)
        if r.error:
            self.ui.note(f"model error: {r.error}", "red")
            sess.feed("Sorry, I lost the model for a second. Say that again.")
        if not r.interrupted and not self.cancel.is_set():
            sess.end_of_text()
        m = sess.wait(timeout=180)
        spoken, unspoken = self._spoken_split(sess, r.text)
        interrupted = r.interrupted or self.cancel.is_set()
        self.brain.commit(turn.user_text, r, spoken if interrupted else r.text, unspoken if interrupted else "")
        self.echo.set_active(False)
        self.session = None
        if not interrupted:
            self.sm.done()

        turn.assistant_text, turn.spoken_text, turn.unspoken_text = r.text, spoken, unspoken if interrupted else ""
        turn.interrupted, turn.tool_calls = interrupted, r.tool_calls
        turn.llm_first_token_ms, turn.llm_first_sentence_ms = r.ttft_ms, r.first_sentence_ms
        turn.tts_first_ms = m.sentences[0].gen_s * 1e3 if m.sentences else None
        if self.turn_sink.first_audio_wall and turn.t_speech_end:
            turn.speech_end_to_audio_ms = (self.turn_sink.first_audio_wall - turn.t_speech_end) * 1e3
        badges = {"asr": f"{turn.asr_ms:.0f}ms" if turn.asr_ms else "-",
                  "llm→sentence": f"{r.first_sentence_ms:.0f}ms" if r.first_sentence_ms else "-",
                  "kokoro": f"{turn.tts_first_ms:.0f}ms" if turn.tts_first_ms else "-",
                  "end→audio": f"{turn.speech_end_to_audio_ms:.0f}ms" if turn.speech_end_to_audio_ms else "-"}
        self.ui.assistant_done(badges)
        self.turns.append(turn)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: v for k, v in turn.__dict__.items() if k != "started"}, ensure_ascii=False) + "\n")

    def _spoken_split(self, sess: StreamingSpeech, full_text: str) -> tuple[str, str]:
        """Which words actually came out of the speaker before the cut."""
        played = self.turn_sink.played_samples - self.session_play0
        spoken_parts, cum = [], 0
        for s in sess.m.sentences:
            n = int(s.audio_s * TTS_SR)
            if cum + n <= played:
                spoken_parts.append(s.text)
            elif played > cum:
                words = s.text.split()
                k = max(1, int(len(words) * (played - cum) / max(n, 1)))
                spoken_parts.append(" ".join(words[:k]))
                break
            else:
                break
            cum += n
        spoken = " ".join(spoken_parts)
        unspoken = full_text[len(spoken):].strip() if full_text.startswith(spoken[:20]) else full_text.replace(spoken, "", 1).strip()
        return spoken, unspoken

    def barge_in(self):
        if self.sm.barge_in():
            self.cancel.set()
            if self.session:
                self.session.interrupt()
            time.sleep(0.03)
            spoken, unspoken = self._spoken_split(self.session, self.session.m.sentences and "".join(s.text + " " for s in self.session.m.sentences) or "") if self.session else ("", "")
            self.ui.interrupted(spoken, unspoken)
            self.sm.resume_user()

    # ---------------------------------------------------------------- typed input (alongside the mic)
    def read_typed(self):
        """Read lines from stdin and inject them as turns, same path as --say. Runs next to the mic
        loop: waits for the agent to be free (or asks for a barge-in) before sending each line."""
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                return
            line = line.strip()
            if not line:
                continue
            deadline = time.perf_counter() + 8
            while self.sm.state != S.LISTENING and time.perf_counter() < deadline:
                time.sleep(0.02)
            if not self.sm.vad_on():
                self.ui.note("(agent busy, dropped typed message — wait for it to finish)", "red dim")
                continue
            self.sm.vad_off()
            self.turn_q.put((None, time.perf_counter(), line))

    # ---------------------------------------------------------------- worker: ASR + respond
    def worker(self):
        while True:
            item = self.turn_q.get()
            if item is None:
                return
            audio, t_end, text = item
            if text is None:
                t0 = time.perf_counter()
                with self.asr_lock:
                    text = self.asr.recognize(audio).strip()
                asr_ms = (time.perf_counter() - t0) * 1e3
            else:
                asr_ms = None
            if not text:
                self.ui.note("(nothing recognised)")
                self.sm.done()
                continue
            self.turn_n += 1
            turn = Turn(self.turn_n, user_text=text, t_speech_end=t_end, asr_ms=asr_ms)
            self.ui.user(turn.n, text, asr_ms)
            self.respond(turn)

    # ---------------------------------------------------------------- mic loop
    def run(self, chunks_source):
        """chunks_source: iterator of float32 16 kHz chunks (real mic or simulation)."""
        from silero_vad import load_silero_vad

        a = self.a
        vad = load_silero_vad()
        threading.Thread(target=self.worker, daemon=True).start()
        preroll = collections.deque(maxlen=int(a.preroll * SR / CHUNK) + 1)
        utt: list[np.ndarray] = []
        t_on = t_last = 0.0
        hot = 0
        last_partial_len = 0
        partial_busy = threading.Event()

        def partial_job(audio: np.ndarray):
            try:
                if self.asr_lock.acquire(timeout=0.05):
                    try:
                        text = self.asr.recognize(audio[-SR * 15:]).strip()
                    finally:
                        self.asr_lock.release()
                    if self.sm.state in (S.USER_SPEAKING, S.INTERRUPTED):
                        self.ui.set_partial(text)
            finally:
                partial_busy.clear()

        keys = None
        if sys.stdin.isatty() and not a.sim:
            keys = LineReader(on_line=self.on_command_line, on_change=self.ui.set_input)
            keys.start()
        for chunk in chunks_source:
            if self.quit.is_set():
                break
            now = time.perf_counter()
            p = vad(torch.from_numpy(chunk), SR).item()
            echo = self.echo.is_echo(chunk) if self.sm.agent_busy else False
            self.ui.set_vad(p, self.echo.last_corr if self.sm.agent_busy else 0.0)
            st = self.sm.state

            if st in (S.LISTENING, S.THINKING, S.SPEAKING):
                preroll.append(chunk)
                thr = a.bargein_threshold if st != S.LISTENING else a.threshold
                if p >= thr and not echo:
                    hot += 1
                    if st == S.LISTENING or hot >= a.bargein_chunks:
                        if st != S.LISTENING:
                            self.barge_in()
                            if self.sm.state != S.USER_SPEAKING:
                                continue
                        else:
                            self.sm.vad_on()
                        t_on = t_last = now
                        utt = list(preroll)
                        last_partial_len = 0
                        hot = 0
                else:
                    hot = 0
            elif st in (S.USER_SPEAKING, S.INTERRUPTED):
                utt.append(chunk)
                if p >= a.threshold:
                    t_last = now
                total = len(utt) * CHUNK
                if (a.say is None and a.partials and total - last_partial_len >= int(a.partial_every * SR)
                        and not partial_busy.is_set()):
                    last_partial_len = total
                    partial_busy.set()
                    threading.Thread(target=partial_job, args=(np.concatenate(utt),), daemon=True).start()
                if (now - t_last) * 1e3 >= a.min_silence or (now - t_on) >= a.max_utt:
                    audio = np.concatenate(utt)
                    if (t_last - t_on) * 1e3 >= a.min_speech:
                        self.sm.vad_off()
                        self.turn_q.put((audio, t_last, None))
                    else:
                        self.sm.blip()
                    preroll.clear()
                    vad.reset_states()
                    hot = 0
        if keys:
            keys.stop()
        # source exhausted: let the last turn finish
        deadline = time.perf_counter() + 120
        while (self.sm.agent_busy or not self.turn_q.empty()) and time.perf_counter() < deadline:
            time.sleep(0.05)

    # ---------------------------------------------------------------- sources
    def mic_chunks(self):
        import sounddevice as sd

        a = self.a
        dev = resolve_input_device(a.in_device)
        q: queue.Queue = queue.Queue()
        pending = np.zeros(0, dtype=np.float32)
        t0 = time.perf_counter()
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, device=dev,
                            extra_settings=wasapi_settings(dev), callback=lambda d, f, t, s: q.put(d[:, 0].copy())):
            while True:
                if a.duration and time.perf_counter() - t0 > a.duration:
                    return
                pending = np.concatenate([pending, q.get()])
                while len(pending) >= CHUNK:
                    yield pending[:CHUNK]
                    pending = pending[CHUNK:]

    def sim_chunks(self):
        """Real-time replay: utterance wav, silence, and (optionally) a second wav injected while the agent speaks."""
        a = self.a
        first = load_wav_16k(Path(a.sim))
        barge = load_wav_16k(Path(a.sim_bargein)) if a.sim_bargein else None
        stream = np.concatenate([np.zeros(SR // 2, dtype=np.float32), first, np.zeros(SR, dtype=np.float32)])
        pos, injected, t_begin = 0, False, time.perf_counter()
        i = 0
        while True:
            if a.duration and time.perf_counter() - t_begin > a.duration:
                return
            target = t_begin + (i + 1) * CHUNK / SR
            while time.perf_counter() < target:
                time.sleep(0.002)
            i += 1
            if pos + CHUNK <= len(stream):
                chunk = stream[pos:pos + CHUNK]
                pos += CHUNK
            else:
                chunk = np.zeros(CHUNK, dtype=np.float32)
            if barge is not None and not injected and self.sm.state == S.SPEAKING and self.sink.first_audio_wall \
                    and time.perf_counter() - self.sink.first_audio_wall >= a.sim_bargein_after:
                injected = True
                stream = np.concatenate([barge, np.zeros(SR * 2, dtype=np.float32)])
                pos = 0
                self.ui.note("sim: injecting second utterance over the agent", "magenta")
            yield chunk
            if pos >= len(stream) and (injected or barge is None) and not self.sm.agent_busy and self.turn_q.empty() \
                    and self.sm.state == S.LISTENING and i * CHUNK / SR > 3 and self.turns and (barge is None or injected):
                if time.perf_counter() - self.turns[-1].started > 2:
                    return

    # ---------------------------------------------------------------- summary
    def summary(self):
        if not self.turns:
            return
        self.ui.note(f"session: {len(self.turns)} turns, {sum(t.interrupted for t in self.turns)} interrupted", "bold")
        for k, label in [("speech_end_to_audio_ms", "speech end → first audio"), ("llm_first_sentence_ms", "LLM first sentence"),
                         ("asr_ms", "ASR"), ("tts_first_ms", "Kokoro first sentence")]:
            v = [getattr(t, k) for t in self.turns if getattr(t, k)]
            if v:
                self.ui.note(f"{label:26} p50 {pct(v, 50):5.0f} ms   p95 {pct(v, 95):5.0f} ms")
        self.ui.note(f"echo gate suppressed {self.echo.suppressed} chunks · log {self.log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic:claude-haiku-4-5")
    ap.add_argument("--agent-name", default="Yeti")
    ap.add_argument("--user-name", default="Nasan")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--first-min-words", type=int, default=6)
    ap.add_argument("--in-device", default="Yeti")
    ap.add_argument("--out-device", default="Yeti")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bargein-threshold", type=float, default=0.75)
    ap.add_argument("--bargein-chunks", type=int, default=4, help="consecutive 32 ms VAD hits to interrupt (4 = 128 ms)")
    ap.add_argument("--echo-corr", type=float, default=0.45, help="echo-gate correlation above which mic = agent's own voice")
    ap.add_argument("--min-silence", type=int, default=600)
    ap.add_argument("--min-speech", type=int, default=250)
    ap.add_argument("--preroll", type=float, default=1.0)
    ap.add_argument("--max-utt", type=float, default=30.0)
    ap.add_argument("--partials", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--partial-every", type=float, default=0.6, help="seconds of new audio between live partials")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--typed", action=argparse.BooleanOptionalAction, default=True,
                     help="also read lines from stdin and send them as turns, alongside the mic")
    ap.add_argument("--plain", action="store_true", help="no live screen; print one line per event (logs, CI)")
    ap.add_argument("--search", action=argparse.BooleanOptionalAction, default=True, help="web tools on at start (starts SearXNG)")
    ap.add_argument("--say", nargs="*", default=None, help="text turns, no mic/ASR")
    ap.add_argument("--sim", default=None, help="wav to replay as the mic")
    ap.add_argument("--sim-bargein", default=None, help="wav injected while the agent speaks")
    ap.add_argument("--sim-bargein-after", type=float, default=1.2)
    a = ap.parse_args()

    t = Talk(a)
    with t.ui:
        try:
            t.load()
            if a.say is not None:
                threading.Thread(target=t.worker, daemon=True).start()
                for text in a.say:
                    t.on_command_line(text)   # "/model ..." works here too; plain text becomes a turn
                    while t.sm.agent_busy or not t.turn_q.empty():
                        time.sleep(0.05)
                    time.sleep(0.2)
                t.turn_q.put(None)
            elif a.sim:
                t.run(t.sim_chunks())
            else:
                if a.typed:
                    threading.Thread(target=t.read_typed, daemon=True).start()
                    t.ui.note("type a line + Enter to send text instead of speaking", "dim italic")
                t.run(t.mic_chunks())
        except KeyboardInterrupt:
            pass
        finally:
            try:
                t.sink.wait_drained(10)
                t.sink.close()
            except Exception:  # noqa: BLE001
                pass
            t.summary()


if __name__ == "__main__":
    main()
