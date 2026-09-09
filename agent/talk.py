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

Mic mute: the button in the Textual UI, F2, or "/mic" stops the mic from turning into turns — nothing
you say reaches the model, the log, or memory while muted. It does not stop the agent's own voice or
touch anything else. Say the wake word (the agent's name by default, e.g. "Yeti" or "hey Yeti") to
un-mute; that one phrase is still checked locally so it can hear you say it — everything else is dropped
unheard.
"""
from __future__ import annotations

import os

# HF_HOME is persisted as a Windows *User* env var pointing at D:\hf-cache (C: doesn't have room for the
# model caches — see README). User env vars only reach NEW processes started from a shell that itself
# started after the var was set; a long-lived terminal tab predating that misses it silently, and the
# fallback (~/.cache/huggingface on C:) re-downloads the ~2.4 GB Parakeet model every launch — which is
# exactly what "loading Parakeet" hanging for a minute+ with no visible progress (stderr is redirected to
# a log file in TUI mode) turned out to be. Setting it here removes the dependency on shell freshness
# entirely; setdefault so an already-correct environment is left alone. Must run before any import
# (kokoro, onnx_asr, ...) that touches huggingface_hub.
if os.name == "nt" and os.path.isdir(r"D:\hf-cache\huggingface"):
    os.environ.setdefault("HF_HOME", r"D:\hf-cache\huggingface")

import argparse
import collections
import json
import queue
import re
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
from endpoint import Endpointer  # noqa: E402
from memory import Memory, make_tools as make_memory_tools  # noqa: E402
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

MEMORY_PROMPT_ADDON = """
Memory
- You have remember/recall/search_history/forget. Save things the user asks you to remember, and preferences they state clearly. A user message may end with a bracketed [memory ...] block: those are hints pulled from past sessions, use them if relevant and ignore them if not; never read the block out loud.
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


FILLERS = ["Still looking.", "One sec.", "Almost there.", "Bear with me."]


class Talk:
    def __init__(self, a: argparse.Namespace, ui=None):
        self.a = a
        self.ui = ui or TerminalUI(model=a.model, voice=a.voice, plain=a.plain)
        self.last_paste: str | None = None
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
        self.mem = Memory()
        self.filler_samples = 0
        self.filler_i = 0
        self.backchannels = 0
        # mic mute: stops turning your speech into turns; agent's own voice and everything else keeps working
        self.mic_muted = False
        self._need_reset = False    # tells the mic loop to drop any half-captured utterance on the next chunk
        self._wake_busy = threading.Event()
        self._wake_speaking = False
        self._wake_utt: list[np.ndarray] = []
        self._wake_t_on = self._wake_t_last = 0.0
        self.vad = None             # set once run() creates the Silero instance; reset_states() on toggle
        self.wake_words = [w.lower() for w in (a.wake_word or [])] or \
            [a.agent_name.lower(), f"hey {a.agent_name.lower()}"]
        self.verbose = a.verbose    # routine per-turn telemetry (memory hints, turn-end reason, backchannels)
        self._last_partial_text = ""  # most recent live caption, used by the Smart Turn filler-word guard
        self.tts_engine = a.tts_engine
        self._tts_cache: dict[str, object] = {}  # engines already loaded this session, kept resident once loaded

    # ---------------------------------------------------------------- loading
    def load(self):
        a = self.a
        self.ui.set_status("loading Parakeet…")
        if a.say is None and not a.no_mic:
            self.asr, _, _ = load_parakeet("hybrid")
            self.asr.recognize(np.zeros(SR, dtype=np.float32))
            assert_cuda_still_active(self.asr, "hybrid")
        self.ui.set_status("loading Smart Turn…")
        self.ep = Endpointer(min_silence_ms=a.min_silence, max_silence_ms=a.max_silence, use_model=a.smart_turn,
                             threshold=a.turn_threshold, high_threshold=a.turn_high_threshold)
        self.ui.set_status(f"loading {a.tts_engine}…")
        self.tts = self._load_tts_engine(a.tts_engine)
        self._tts_cache[a.tts_engine] = self.tts
        self.sink = NullSink(on_play=self.echo.push_played) if a.no_play else \
            SpeakerSink(device=resolve_output_device(a.out_device), on_play=self.echo.push_played)
        self.ui.set_status("connecting to model…")
        system = (voice_system_prompt(agent_name=a.agent_name, user_name=a.user_name) + SEARCH_PROMPT_ADDON
                  + MEMORY_PROMPT_ADDON + INTERRUPT_PROMPT_ADDON)
        from brain import current_time
        self.brain = Brain(a.model, system_prompt=system, max_tokens=a.max_tokens, search=False,
                           tools=[current_time] + make_memory_tools(self.mem))
        self.brain.on_notice = lambda msg: self.ui.note(msg, "yellow")
        self.brain.warm()
        if a.search:
            self.ui.note(self._search_on(), "dim")
        for msg in self.brain.mcp_autoconnect():
            self.ui.note(f"mcp: {msg}", "yellow dim")
        self.ui.set_status("")
        self.ui.note(f"ready · Parakeet hybrid · {a.tts_engine} {self.tts.voice} ({self.tts.vram_mib():.0f} MiB) · "
                     f"{a.model} · tools: {', '.join(n for n, _ in self.brain.tool_sources())} · "
                     f"mic mute: button/F2/\"{self.wake_words[0]}\" · /tts neutts for cloning · type /help", "green dim")

    def _load_tts_engine(self, name: str, voice: str | None = None):
        a = self.a
        if name == "kokoro":
            return KokoroTTS(voice=voice or a.voice)
        sys.path.insert(0, str(ROOT / "tts"))
        from neutts_stream import NeuTTSEngine

        return NeuTTSEngine(voice=voice or a.neutts_voice, backbone=a.neutts_backbone)

    def _switch_tts(self, name: str) -> None:
        """Swap the live TTS engine. Each engine loads once and stays resident for the rest of the
        session (switching back is instant); the first switch to neutts pays its ~1-2 min load cost."""
        if name in self._tts_cache:
            self.tts = self._tts_cache[name]
        else:
            self.tts = self._load_tts_engine(name)
            self._tts_cache[name] = self.tts
        self.tts_engine = name

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
            presets = "\n".join(f"/{k:<10} {v[:70]}…" for k, v in self.PRESETS.items())
            return r.help_text() + "\n\npresets (after a paste, or with the text under the command):\n" + presets

        @r.add("model", "model <spec|fast|smart|opus|sonnet|gemini|qwen>", "switch the model (fast = Groq gpt-oss-20b, smart = Haiku 4.5)")
        def _model(args):
            if not args:
                return f"model is {self.brain.model_spec}"
            self.ui.set_status("switching model…")
            try:
                spec = self.brain.set_model(args[0])
                self.ui.model = spec  # must land before the set_status("") below repaints the header,
                # or the status bar keeps showing the old model even though the switch already happened
            finally:
                self.ui.set_status("")
            return f"model → {spec}"

        @r.add("voice", "voice <name>", "switch the voice: a Kokoro name (af_bella, ...) or, on NeuTTS, a cloned reference")
        def _voice(args):
            if not args:
                return f"voice is {self.tts.voice} ({self.tts_engine})"
            self.ui.set_status("switching voice…")
            try:
                self.tts.set_voice(args[0])
                self.ui.voice = args[0]  # before set_status("") repaints the header below, not after
            finally:
                self.ui.set_status("")
            return f"voice → {args[0]}"

        @r.add("tts", "tts [kokoro|neutts]", "switch the TTS engine (neutts loads lazily, ~1-2 min first time)")
        def _tts(args):
            if not args:
                return f"engine is {self.tts_engine} · voice {self.tts.voice}"
            if args[0] not in ("kokoro", "neutts"):
                return "usage: /tts [kokoro|neutts]"
            if args[0] == self.tts_engine:
                return f"already on {args[0]}"
            self.ui.set_status(f"loading {args[0]}…")
            try:
                self._switch_tts(args[0])
                self.ui.voice = self.tts.voice  # before set_status("") repaints the header below
            except Exception as e:  # noqa: BLE001
                return f"could not switch to {args[0]}: {type(e).__name__}: {str(e)[:200]}"
            finally:
                self.ui.set_status("")
            return f"tts → {args[0]} (voice {self.tts.voice})"

        @r.add("clone", "clone <wav>[|transcript] | <name>", "clone a voice from a reference clip and switch NeuTTS to speak as it")
        def _clone(args):
            if not args:
                from neutts_stream import bundled_voices
                return f"bundled: {', '.join(bundled_voices()) or 'none downloaded'} · usage: /clone path.wav[|transcript text]"
            spec = " ".join(args)
            self.ui.set_status("cloning voice…")
            try:
                if self.tts_engine != "neutts":
                    self._switch_tts("neutts")
                self.tts.set_voice(spec)
                self.ui.voice = self.tts.voice  # before set_status("") repaints the header below
            except Exception as e:  # noqa: BLE001
                return f"clone failed: {type(e).__name__}: {str(e)[:200]}"
            finally:
                self.ui.set_status("")
            return f"cloned and switched to {self.tts.voice} (say something to hear it: /say hello there)"

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
            self._say(text)
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

        @r.add("verbose", "verbose [on|off]", "show routine per-turn telemetry (memory hints, turn-end reason, backchannels) inline")
        def _verbose(args):
            if not args:
                return f"verbose is {'on' if self.verbose else 'off'}"
            self.verbose = args[0] == "on"
            return f"verbose {'on' if self.verbose else 'off'}"

        @r.add("mic", "mic [on|off|toggle]", "mute the microphone — stops it listening to you; wake word or the button/F2 turns it back on")
        def _mic(args):
            choice = args[0] if args else "toggle"
            if choice not in ("on", "off", "toggle"):
                return "usage: /mic [on|off|toggle]"
            self.toggle_mic(True if choice == "off" else False if choice == "on" else None)
            return None  # toggle_mic already posts its own note

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
            return (f"state {self.sm.state.value} · mic {'MUTED' if self.mic_muted else 'on'} · model {self.brain.model_spec} · "
                    f"tts {self.tts_engine} voice {self.tts.voice} ({', '.join(self._tts_cache)} loaded) · "
                    f"search {'on' if self.brain.search_on else 'off'} · mcp {', '.join(self.brain.mcp_tools) or 'none'} · "
                    f"{'muted' if self.muted else 'sound on'} · echo gate suppressed {self.echo.suppressed}\n  {lt}")

        @r.add("memory", "memory [search <q> | notes | forget <id> | stats]", "the SQLite memory (cheap RAG)")
        def _memory(args):
            if not args or args[0] == "stats":
                st = self.mem.stats()
                return f"{st['turns']} turns, {st['notes']} notes, {st['sessions']} sessions · {st['db']}"
            if args[0] == "notes":
                return "\n".join(f"  #{n['id']} ({n['when']}) {n['text']}" for n in self.mem.recent_notes(10)) or "  (no notes)"
            if args[0] == "search" and len(args) > 1:
                q = " ".join(args[1:])
                rows = self.mem.search_notes(q, 5) + self.mem.search_turns(q, 5)
                return "\n".join(f"  {x.get('role', 'note'):5} ({x['when']}) {x['text'][:120]}" for x in rows) or "  nothing"
            if args[0] == "forget" and len(args) > 1:
                return "deleted" if self.mem.forget(int(args[1])) else "no such note"
            return "usage: /memory [search <q> | notes | forget <id> | stats]"

        @r.add("quit", "quit", "exit")
        def _quit(args):
            self.quit.set()
            if hasattr(self.ui, "quit_app"):
                self.ui.quit_app()  # TextualUI: close the screen too, not just the mic/agent loop
            return "bye"

        return r

    PRESETS = {
        "explain": "Explain what this is and what it does, in plain spoken language. Lead with the one-sentence version, then the two or three things that matter most.",
        "review": "Review this. Tell me the most important problem first, then anything else worth fixing. Be direct.",
        "next": "Given this, what should I do next? Give me the single best next step and why, then a fallback.",
        "summarize": "Summarize this in a few spoken sentences.",
        "fix": "Something is wrong with this. Find the most likely cause and tell me the fix.",
        "why": "Why is this happening? Walk me through the cause.",
    }

    def submit_text(self, line: str) -> None:
        """Everything typed or pasted comes through here: slash commands, presets, pastes, plain turns."""
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            return
        first, _, rest = stripped.partition("\n")
        head = first.strip()
        # preset on the first line (optionally with extra words), body = paste or the last paste
        if head.startswith("/") and head[1:].split(" ")[0] in self.PRESETS:
            name, _, extra = head[1:].partition(" ")
            body = rest.strip() or (self.last_paste or "")
            instruction = self.PRESETS[name] + (f" {extra.strip()}" if extra.strip() else "")
            if not body:
                self.ui.note(f"/{name}: paste something first (or put it under the command)", "yellow")
                return
            self._paste_turn(instruction, body)
            return
        if head.startswith("/"):
            out = self.router.dispatch(head)
            if out:
                for ln in str(out).split("\n"):
                    self.ui.note(ln, "bold" if ln.startswith("/") else "white")
            return
        if "\n" in stripped or len(stripped) > 400:
            # a paste: keep it as context and ask what it is
            self._paste_turn("I just pasted this. Tell me briefly what it is and what it does, then keep it in mind for follow-up questions.", stripped)
            return
        self._say(stripped)

    def _say(self, text: str, model_text: str | None = None) -> None:
        if self.sm.agent_busy:
            self.barge_in()
            self.sm.abort()
        self.sm.vad_on()
        self.sm.vad_off()
        self.turn_q.put((None, time.perf_counter(), text, model_text))

    def _paste_turn(self, instruction: str, body: str) -> None:
        self.last_paste = body
        n_lines = body.count("\n") + 1
        self.turn_n_preview = self.turn_n + 1
        first_line = body.strip().split("\n")[0][:100]
        if hasattr(self.ui, "paste"):
            self.ui.paste(self.turn_n + 1, first_line, n_lines, instruction)
        model_text = f"{instruction}\n\n```\n{body}\n```"
        self._say(f"{instruction} [pasted {n_lines} lines: {first_line}]", model_text)

    def on_command_line(self, line: str) -> None:
        self.submit_text(line)

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
        self.filler_samples = 0
        self.session_play0 = sink.played_samples
        sink.finished = False
        sink.first_audio_wall = None
        self.echo.set_active(not self.muted)

        def on_text(d):
            sess.feed(d)
            self.ui.assistant_text(d)
            if self.turn_sink.first_audio_wall and self.sm.state == S.THINKING:
                self.sm.first_audio()

        tool_pending = {"n": 0}

        def filler_after(delay: float, tool_n: int):
            """While the tool runs: if the speaker has been silent for `delay` s, say one short filler."""
            silent_since = None
            while tool_pending["n"] == tool_n and not self.cancel.is_set():
                if self.muted:
                    return
                if sink.queued_samples() > 0:
                    silent_since = None
                else:
                    silent_since = silent_since or time.perf_counter()
                    if time.perf_counter() - silent_since >= delay:
                        phrase = FILLERS[self.filler_i % len(FILLERS)]
                        self.filler_i += 1
                        audio = self.tts.synth(phrase, a.speed)
                        if tool_pending["n"] == tool_n and not self.cancel.is_set():
                            self.filler_samples += len(audio)
                            sink.push(audio)
                            if self.verbose:
                                self.ui.note(f"filler: {phrase}", "dim")
                        return
                time.sleep(0.1)

        def on_tool_call(name):
            sess.flush()  # say the lead-in phrase now, not after the tool returns
            self.ui.tool(name)
            tool_pending["n"] += 1
            threading.Thread(target=filler_after, args=(a.filler_after, tool_pending["n"]), daemon=True).start()

        def on_tool_result(name, res):
            tool_pending["n"] += 1  # invalidates the pending filler
            self.ui.tool(name, res)

        # watch for first audio while the LLM is still streaming (audio can start before the next delta)
        def watch_audio():
            while self.session is sess and not sess.m.t_done:
                if self.turn_sink.first_audio_wall and self.sm.state == S.THINKING:
                    self.sm.first_audio()
                    break
                time.sleep(0.01)
        threading.Thread(target=watch_audio, daemon=True).start()

        ctx = self.mem.context_for(turn.user_text)
        base = turn.user_text_for_model or turn.user_text
        model_text = base + ("\n" + ctx if ctx else "")
        if ctx and self.verbose:
            self.ui.note(f"memory: {ctx.count(chr(10)) - 1} hint(s) attached", "dim")
        r = self.brain.stream_turn(model_text, on_text=on_text, on_tool_call=on_tool_call,
                                   on_tool_result=on_tool_result, cancel=self.cancel)
        turn.user_text_for_model = model_text
        if r.error:
            self.ui.note(f"model error: {r.error}", "red")
            sess.feed("Sorry, I lost the model for a second. Say that again.")
        if not r.interrupted and not self.cancel.is_set():
            sess.end_of_text()
        m = sess.wait(timeout=180)
        spoken, unspoken = self._spoken_split(sess, r.text)
        interrupted = r.interrupted or self.cancel.is_set()
        self.brain.commit(turn.user_text_for_model, r, spoken if interrupted else r.text, unspoken if interrupted else "")
        self.mem.log_turn("user", turn.user_text)
        self.mem.log_turn("agent", spoken if interrupted else r.text, interrupted)
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
        played = self.turn_sink.played_samples - self.session_play0 - self.filler_samples
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

    # ---------------------------------------------------------------- mic mute + wake word
    def toggle_mic(self, muted: bool | None = None) -> None:
        """muted=True mutes (stop listening), False unmutes, None flips it. Safe from any thread."""
        new_state = (not self.mic_muted) if muted is None else muted
        if new_state == self.mic_muted:
            return
        self.mic_muted = new_state
        self._need_reset = True
        self._wake_reset()
        if self.vad is not None:
            self.vad.reset_states()
        if new_state:
            self.sm.force_listening("mic_muted")  # never touches THINKING/SPEAKING — the reply keeps going
        if hasattr(self.ui, "set_mic"):
            self.ui.set_mic(new_state)
        self.ui.note(f"mic muted — say \"{self.wake_words[0]}\", press the button, or hit F2 to resume" if new_state
                     else "mic on — listening again", "yellow" if new_state else "green")

    def _wake_reset(self) -> None:
        self._wake_speaking = False
        self._wake_utt = []

    def _wake_step(self, chunk: np.ndarray, vad, now: float) -> None:
        """Runs instead of the normal turn logic while muted: segments speech with a fixed hangover and
        checks each utterance for the wake word. Nothing here reaches the model, the log, or memory."""
        a = self.a
        p = vad(torch.from_numpy(chunk), SR).item()
        self.ui.set_vad(p, 0.0)
        if not self._wake_speaking:
            if p >= a.threshold:
                self._wake_speaking = True
                self._wake_t_on = self._wake_t_last = now
                self._wake_utt = [chunk]
            return
        self._wake_utt.append(chunk)
        if p >= a.threshold:
            self._wake_t_last = now
        if (now - self._wake_t_last) * 1e3 < a.wake_hangover:
            return
        self._wake_speaking = False
        audio = np.concatenate(self._wake_utt)
        self._wake_utt = []
        if (self._wake_t_last - self._wake_t_on) * 1e3 < a.min_speech or self._wake_busy.is_set():
            return
        self._wake_busy.set()
        threading.Thread(target=self._wake_check, args=(audio,), daemon=True).start()

    def _wake_check(self, audio: np.ndarray) -> None:
        try:
            with self.asr_lock:
                text = self.asr.recognize(audio).strip()
            norm = " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
            if any(w in norm for w in self.wake_words):
                self.toggle_mic(False)
                if not self.muted:
                    try:
                        self.sink.push(self.tts.synth("Yeah?", self.a.speed))
                    except Exception:  # noqa: BLE001
                        pass
            elif self.a.wake_debug:
                self.ui.note(f"(muted, heard: \"{text}\" — not the wake word)" if text else "(muted, heard silence)", "dim")
        finally:
            self._wake_busy.clear()

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
            audio, t_end, text = item[0], item[1], item[2]
            model_text = item[3] if len(item) > 3 else None
            if text is None:
                t0 = time.perf_counter()
                with self.asr_lock:
                    text = self.asr.recognize(audio).strip()
                asr_ms = (time.perf_counter() - t0) * 1e3
            else:
                asr_ms = None
            if not text:
                if self.verbose:
                    self.ui.note("(nothing recognised)")
                self.sm.done()
                continue
            self.turn_n += 1
            turn = Turn(self.turn_n, user_text=text, t_speech_end=t_end, asr_ms=asr_ms)
            turn.user_text_for_model = model_text or ""
            if not model_text:
                self.ui.user(turn.n, text, asr_ms)
            self.respond(turn)

    # ---------------------------------------------------------------- mic loop
    def run(self, chunks_source):
        """chunks_source: iterator of float32 16 kHz chunks (real mic or simulation)."""
        from silero_vad import load_silero_vad

        a = self.a
        vad = load_silero_vad()
        self.vad = vad
        threading.Thread(target=self.worker, daemon=True).start()
        preroll = collections.deque(maxlen=int(a.preroll * SR / CHUNK) + 1)
        utt: list[np.ndarray] = []
        t_on = t_last = 0.0
        hot = 0        # leaky accumulator of net sustained-speech credit while the agent may be barged in on
        miss_run = 0   # consecutive silent/below-threshold chunks since the last hit, once ducked
        last_partial_len = 0
        partial_busy = threading.Event()

        def partial_job(audio: np.ndarray):
            try:
                if self.asr_lock.acquire(timeout=0.05):
                    try:
                        text = self.asr.recognize(audio[-SR * 15:]).strip()
                    finally:
                        self.asr_lock.release()
                    self._last_partial_text = text  # feeds the Smart Turn filler-word guard even after this clears
                    if self.sm.state in (S.USER_SPEAKING, S.INTERRUPTED):
                        self.ui.set_partial(text)
            finally:
                partial_busy.clear()

        keys = None
        if sys.stdin.isatty() and not a.sim and not getattr(a, "tui", False):
            keys = LineReader(on_line=self.on_command_line, on_change=self.ui.set_input)
            keys.start()
        ducked = False
        for chunk in chunks_source:
            if self.quit.is_set():
                break
            now = time.perf_counter()
            if self._need_reset:
                self._need_reset = False
                hot = 0
                miss_run = 0
                ducked = False
                self.sink.gain = 1.0
                preroll.clear()
                utt = []
            if self.mic_muted:
                self._wake_step(chunk, vad, now)
                continue
            p = vad(torch.from_numpy(chunk), SR).item()
            echo = self.echo.is_echo(chunk) if self.sm.agent_busy else False
            self.ui.set_vad(p, self.echo.last_corr if self.sm.agent_busy else 0.0)
            st = self.sm.state

            if st in (S.LISTENING, S.THINKING, S.SPEAKING):
                preroll.append(chunk)
                thr = a.bargein_threshold if st != S.LISTENING else a.threshold
                if p >= thr and not echo:
                    miss_run = 0
                    hot = min(hot + 1, a.bargein_chunks + a.duck_chunks)  # cap: no unbounded credit build-up
                    if st == S.LISTENING:
                        self.sm.vad_on()
                        self.ep.start_utterance()
                        t_on = t_last = now
                        utt = list(preroll)
                        last_partial_len = 0
                        hot = 0
                    else:
                        # two-stage barge-in: duck first (a backchannel "mhm" passes), cut only once sustained
                        # speech has accumulated enough NET credit — a lone missed 32 ms frame mid-sentence
                        # (a natural VAD dip) only costs --bargein-decay, it does not wipe the count to zero,
                        # so real continuous speech reliably reaches the cut threshold even with flutter.
                        if hot >= a.duck_chunks and not ducked:
                            ducked = True
                            self.sink.gain = a.duck_gain
                        if hot >= a.bargein_chunks:
                            self.sink.gain = 1.0
                            ducked = False
                            hot = 0
                            miss_run = 0
                            self.barge_in()
                            if self.sm.state != S.USER_SPEAKING:
                                continue
                            self.ep.start_utterance()
                            t_on = t_last = now
                            utt = list(preroll)
                            last_partial_len = 0
                else:
                    hot = max(0, hot - a.bargein_decay)
                    if ducked:
                        miss_run += 1
                        # only call it a finished backchannel after sustained silence, not the first stray miss —
                        # otherwise natural micro-pauses inside one utterance flicker the volume and spam the log
                        if miss_run >= a.bargein_release_chunks:
                            ducked = False
                            self.sink.gain = 1.0
                            hot = 0
                            miss_run = 0
                            self.backchannels += 1
                            if self.verbose:
                                self.ui.note("backchannel ignored", "dim")
                    else:
                        miss_run = 0
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
                sil_ms = (now - t_last) * 1e3
                decision = self.ep.update(np.concatenate(utt), sil_ms, self._last_partial_text) \
                    if sil_ms >= self.ep.min_ms else None
                if decision or (now - t_on) >= a.max_utt:
                    audio = np.concatenate(utt)
                    if (t_last - t_on) * 1e3 >= a.min_speech:
                        self.sm.vad_off()
                        if self.verbose:
                            if decision == "done" and self.ep.last_p is not None:
                                self.ui.note(f"turn end: smart-turn {self.ep.last_p:.2f} after {sil_ms:.0f} ms", "dim")
                            elif decision == "timeout":
                                self.ui.note(f"turn end: silence {sil_ms:.0f} ms" + (f" (smart-turn last {self.ep.last_p:.2f})" if self.ep.last_p is not None else ""), "dim")
                        self._last_partial_text = ""
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
    def idle_chunks(self):
        """No microphone: feed silence so the loop (and typed turns) still run."""
        while not self.quit.is_set():
            time.sleep(0.032)
            yield np.zeros(CHUNK, dtype=np.float32)

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
        if a.sim_len:
            first = first[: int(a.sim_len * SR)]
        barge = load_wav_16k(Path(a.sim_bargein)) if a.sim_bargein else None
        if barge is not None and a.sim_bargein_len:
            barge = barge[: int(a.sim_bargein_len * SR)]
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
        self.ui.note(f"echo gate suppressed {self.echo.suppressed} chunks · backchannels ignored {self.backchannels} · "
                     f"smart-turn checks {self.ep.checks} (p50 {pct(self.ep.cost_ms, 50):.0f} ms, "
                     f"{self.ep.held_on_filler} held on a trailing filler word) · log {self.log_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic:claude-haiku-4-5")
    ap.add_argument("--agent-name", default="Yeti")
    ap.add_argument("--user-name", default="Nasan")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice (af_heart, af_bella, am_michael, ...)")
    ap.add_argument("--tts-engine", choices=["kokoro", "neutts"], default="kokoro",
                     help="kokoro (fast, fixed voices) or neutts (slower, clones a reference voice); "
                          "switch live with /tts")
    ap.add_argument("--neutts-voice", default="dave", help="NeuTTS reference: a bundled sample name, "
                     "a wav path (needs a matching .txt), or 'wav_path|transcript'")
    ap.add_argument("--neutts-backbone", default="neuphonic/neutts-air-q8-gguf",
                     help="neuphonic/neutts-air-q8-gguf (default) or -q4-gguf for less VRAM/more speed at lower quality")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--first-min-words", type=int, default=6)
    ap.add_argument("--in-device", default="Yeti")
    ap.add_argument("--out-device", default="Yeti")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bargein-threshold", type=float, default=0.75)
    ap.add_argument("--bargein-chunks", type=int, default=10,
                     help="net accumulated 32 ms hits of sustained speech to cut the agent (10 = 320 ms of speech "
                          "credit; a brief VAD dip only costs --bargein-decay units, not a full reset)")
    ap.add_argument("--duck-chunks", type=int, default=3, help="net hits before the voice is ducked (3 = 96 ms)")
    ap.add_argument("--duck-gain", type=float, default=0.25)
    ap.add_argument("--bargein-decay", type=int, default=1,
                     help="units subtracted from the barge-in credit per missed (below-threshold) chunk; "
                          "1 means a single 32 ms dip mid-sentence barely costs anything, unlike a hard reset")
    ap.add_argument("--bargein-release-chunks", type=int, default=6,
                     help="consecutive silent chunks needed after ducking before it's called a finished "
                          "backchannel and the volume is restored (6 = ~190 ms); avoids repeated duck/undock "
                          "flicker and repeated 'backchannel ignored' notes during one continuous utterance")
    ap.add_argument("--smart-turn", action=argparse.BooleanOptionalAction, default=True, help="Smart Turn v3.2 end-of-turn model")
    ap.add_argument("--turn-threshold", type=float, default=0.7,
                     help="minimum Smart Turn score to even consider the turn finished")
    ap.add_argument("--turn-high-threshold", type=float, default=0.9,
                     help="Smart Turn score at/above which the turn ends even on a trailing filler word "
                          "('um', 'so', 'and', ...); between --turn-threshold and this, a filler-ending "
                          "partial holds the turn open instead of cutting mid-hedge")
    ap.add_argument("--max-silence", type=int, default=1200, help="hard end of turn after this much silence (ms)")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                     help="show routine per-turn telemetry (memory hints, turn-end reason, backchannels) inline "
                          "instead of keeping the conversation view to just what was said")
    ap.add_argument("--filler-after", type=float, default=1.5, help="seconds a tool may run silently before a filler phrase")
    ap.add_argument("--sim-len", type=float, default=0, help="use only the first N s of --sim")
    ap.add_argument("--sim-bargein-len", type=float, default=0, help="use only the first N s of --sim-bargein (0.3 = a backchannel)")
    ap.add_argument("--echo-corr", type=float, default=0.45, help="echo-gate correlation above which mic = agent's own voice")
    ap.add_argument("--min-silence", type=int, default=250, help="silence before Smart Turn starts scoring (ms)")
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
    ap.add_argument("--tui", action=argparse.BooleanOptionalAction, default=None, help="Textual screen with the multi-line editor (default on a terminal)")
    ap.add_argument("--no-mic", action="store_true", help="text only: no microphone, no Parakeet; replies are still spoken")
    ap.add_argument("--search", action=argparse.BooleanOptionalAction, default=True, help="web tools on at start (starts SearXNG)")
    ap.add_argument("--wake-word", nargs="*", default=None,
                     help="phrase(s) that un-mute the mic while muted (default: the agent's name, and 'hey <name>')")
    ap.add_argument("--wake-hangover", type=int, default=500, help="silence (ms) that closes an utterance while checking for the wake word")
    ap.add_argument("--wake-debug", action="store_true", help="show everything heard while muted, not just wake-word hits")
    ap.add_argument("--say", nargs="*", default=None, help="text turns, no mic/ASR")
    ap.add_argument("--sim", default=None, help="wav to replay as the mic")
    ap.add_argument("--sim-bargein", default=None, help="wav injected while the agent speaks")
    ap.add_argument("--sim-bargein-after", type=float, default=1.2)
    return ap


def main():
    a = build_parser().parse_args()

    if a.tui is None:
        a.tui = sys.stdin.isatty() and not a.plain and a.say is None and not a.sim
    if a.tui:
        from tui import TextualUI, run_app

        ui = TextualUI(model=a.model, voice=a.voice)
        t = Talk(a, ui=ui)
        # Textual owns the terminal: keep library chatter (torch warnings, ORT) out of the screen
        sys.stderr = open(RESULTS_DIR / "talk_stderr.log", "a", encoding="utf-8")

        def start():
            t.load()
            t.run(t.idle_chunks() if a.no_mic else t.mic_chunks())

        try:
            run_app(ui, start, t.submit_text, mic_fn=t.toggle_mic)
        finally:
            t.quit.set()
            try:
                t.sink.close()
            except Exception:  # noqa: BLE001
                pass
            t.summary()
        return

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
