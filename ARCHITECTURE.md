# voiceAgent — Architecture

## 1. High-level overview

voiceAgent is a native-Windows, real-time voice assistant: a continuous mic loop does local speech
detection and transcription, hands finished utterances to a LangGraph tool-calling agent backed by a
pluggable LLM provider, and speaks the streamed reply back sentence-by-sentence through a resident TTS
engine — all while watching for the user talking over it (barge-in) and its own voice leaking back into
the mic (echo). It is a collection of standalone, independently runnable Python scripts/venvs (STT
benchmark, TTS benchmark, SearXNG server, the agent itself) that converge in `agent/talk.py`, not a single
packaged application.

```
 mic ──▶ sounddevice InputStream (16 kHz, 32 ms chunks)
          │
          ▼
        Silero VAD (per chunk) ──▶ EchoGate (ignore agent's own leaked audio)
          │
          ▼
   StateMachine (state.py): LISTENING → USER_SPEAKING → THINKING → SPEAKING → (INTERRUPTED)
          │                                   ▲
          │  utterance buffer                 │ barge-in (two-stage: duck, then cut)
          ▼                                   │
   Endpointer (endpoint.py): Silero silence + Smart Turn v3.2 ONNX model, 3-tier confidence
          │  "done"
          ▼
   Parakeet (onnx-asr, hybrid CUDA/CPU) ──▶ text
          │
          ▼
   Memory.context_for() (SQLite FTS5) ──▶ prepend "[memory ...]" block
          │
          ▼
   Brain (brain.py): LangGraph create_agent, tools + MCP, streamed token-by-token
          │  text deltas                         │ tool calls
          ▼                                       ▼
   SentenceChunker ──▶ StreamingSpeech ──▶ KokoroTTS | NeuTTSEngine (subprocess) ──▶ SpeakerSink (sounddevice)
                                                                                        │
                                                                                        └─▶ EchoGate reference
```

Everything is orchestrated inside `agent/talk.py`'s `Talk` class, which owns the mic loop, the state
machine, the endpointer, the ASR lock, the Brain instance, the TTS session, and the Rich/Textual UI.

---

## 2. LLM providers (`llm/providers.py`, `llm/prompts.py`, `agent/brain.py`)

**Abstraction layer, not hardcoded per-provider.** There are actually two parallel abstraction layers,
used in different places:

- `llm/providers.py` wraps **LiteLLM** directly: `stream_chat(model, messages, system=..., stats=...)`
  is a plain generator of text deltas, used by the older `agent/voice_agent.py` and by `llm/llm_test.py`
  for benchmarking. Model strings are LiteLLM's `provider/model` form, e.g. `groq/openai/gpt-oss-20b`,
  `anthropic/claude-opus-5`, `gemini/gemini-3.5-flash-lite`.
- `agent/brain.py` (the one `agent/talk.py` actually uses) goes through **LangChain's `init_chat_model`**
  + `langchain.agents.create_agent`, using LangChain's `provider:model` spec form (colon, not slash) —
  `anthropic:claude-haiku-4-5`, `groq:openai/gpt-oss-20b`, `google_genai:gemini-3.5-flash-lite`. This is
  the path that gets LangGraph's streaming/tool-calling graph, so it duplicates some of `providers.py`'s
  concerns (Groq key selection, reasoning-effort knobs) independently rather than reusing it — the only
  cross-import is `from providers import groq_pool`, for the shared key pool.
- **Default model**: `agent/talk.py` defaults to `anthropic:claude-haiku-4-5`. `MODEL_ALIASES` in
  `brain.py` map short names (`fast`, `smart`, `opus`, `sonnet`, `gemini`, `qwen`) to full specs, settable
  live with `/model <alias-or-spec>`.
- **Groq key pool**: `GroqKeyPool` (providers.py) round-robins a JSON list of keys from `GROQ_KEYS`,
  benches a key for 60 s on 429 and permanently on 401/invalid, and has `first_working()` to probe via a
  cheap `GET /models` before committing to a key. `Brain.set_model()` calls this when switching to a Groq
  model and injects the chosen key into `GROQ_API_KEY` for LangChain to pick up (a global-env-var handoff
  between two different SDK integrations — see "fragile" below).
- **Streaming/latency instrumentation** is a first-class concern in both layers: `StreamStats` /
  `TurnResult` track time-to-first-token, time-to-first-sentence (via a sentence-end regex), total tokens,
  because these numbers directly determine how "alive" the spoken agent feels.
- **Overload fallback**: `Brain.stream_turn()` retries once on the same model for transient errors
  (529/503/502/timeout), then falls back to the `fast` alias for exactly one turn before returning to the
  user's chosen model.

**Fragile / worth reconsidering:**
- Two independent provider abstractions (LiteLLM-direct vs LangChain-`init_chat_model`) that don't share
  code beyond the Groq key pool — a change to model defaults or provider quirks has to be made twice.
- Groq key rotation crosses an SDK boundary via a mutated `os.environ["GROQ_API_KEY"]` global — works for
  one agent process but is not thread-safe if this were ever parallelized.
- `provider_extra()` / the `reasoning_effort` hacks are model-name string-matching (`model.startswith(...)`)
  that will silently stop applying if a model is renamed upstream.

---

## 3. Prompts (`llm/prompts.py`)

There is **one** system prompt, `VOICE_SYSTEM_PROMPT` (a Python format-string template filled with
`agent_name`/`user_name`), used for every turn regardless of model or context. It is entirely tuned for
*speech*: explicit rules like "no markdown, no bullet points, no headings, no emojis, no code blocks, no
URLs", "say numbers/dates the way they're spoken aloud", "answer first in one or two sentences."

**There is no separate response-style switch** (no short-vs-comprehensive or plain-vs-markdown prompt
variants, no per-mode system prompt selection). What looks like style switching is actually two different
mechanisms:
1. **Presets** (`agent/talk.py: Talk.PRESETS` — `explain`, `review`, `next`, `summarize`, `fix`, `why`) are
   canned *instruction text* prepended to a pasted-block turn (e.g. `/review` on a pasted code snippet).
   They still go through the same single voice system prompt and are explicitly worded to ask for "plain
   spoken language" — they change what is asked, not the rules for how it must sound.
2. The Textual UI (`agent/tui.py`) can display arbitrarily long/rich text on screen (it detects and
   preserves code fences on paste), but what is *spoken* is still governed by the one voice prompt.

**Fragile / worth reconsidering:** if a future mode genuinely needs markdown/code output (e.g. a "text
chat, not spoken" mode), there is currently no second prompt or flag to switch to — the single
speech-tuned prompt would need a real branch, not just a preset instruction.

---

## 4. Web/search tool (`search/`, `agent/tools_web.py`)

**Mechanics:** SearXNG runs from source, not Docker, in its own venv:
- `search/install_searxng.py` downloads the SearXNG GitHub master zip, extracts to `search/searxng-src`
  (skipping 4 filenames illegal on NTFS), regex-patches Unix-only imports (`pwd`, `grp`, `fcntl`,
  `resource`, `termios`, `pty`) to no-op on import failure, then creates `.venv-searxng` (uv, Python 3.11)
  and `pip install`s SearXNG's own `requirements.txt` into it.
- `search/run_searxng.py` starts it as a subprocess: `python -m searx.webapp`, cwd
  `search/searxng-src`, env `SEARXNG_SETTINGS_PATH=search/settings.yml`. `ensure_running()` is what
  `agent/talk.py`'s `/search on` calls — it starts the process if `GET {URL}/healthz` isn't already 200,
  and polls up to 25 s. Logs go to `results/searxng.log`.
- **URL/port**: `http://127.0.0.1:8888` by default (`SEARXNG_URL` env var overrides), read independently
  by both `search/run_searxng.py` and `agent/tools_web.py`.
- **What must be running**: nothing external — SearXNG itself is the whole dependency; it fans out to
  public search engines (Google, Brave, Startpage, Wikipedia, GitHub, StackOverflow, arXiv) over the
  network. DuckDuckGo/Qwant/Bing are disabled in `search/settings.yml` (captcha from this IP / weak
  results).
- `agent/tools_web.py` provides three LangChain tools on top: `web_search` (SearXNG JSON API, top 6 hits
  + answers/infobox), `read_page` (Trafilatura extraction with a PyPI-JSON-API special case since PyPI
  blocks scrapers, trimmed to 6k chars), `research` (search then read top N pages in parallel via
  `ThreadPoolExecutor`).

**Packaging for a public repo — exclude:**
- `search/searxng-src/` (already `.gitignore`d) — it's fetched by `install_searxng.py`, not source here.
- `.venv-searxng/`, `.venv/`, `.env` (already gitignored).
- `search/settings.yml`'s `secret_key` is a static, checked-in Flask session-signing key. It's a
  meaningless placeholder for a localhost-only single-user instance, but a public template should either
  regenerate it on install or clearly comment that it must be replaced.
- `agent/mcp.json`'s disabled `filesystem` server hardcodes an absolute local path
  (`D:\\Downloads\\voiceAgent`) — needs to become a relative/`${ROOT}`-substituted path (the code already
  supports `${ROOT}`/`${VENV_PYTHON}` substitution for `command`/`args`, just not applied to this one).
- No API keys are involved in the SearXNG path itself (it's not calling paid search APIs).

**Fragile / worth reconsidering:** SearXNG's Flask dev server is explicitly "fine for one user on
localhost" per its own docstring — not for anything beyond that. The install script patches SearXNG's
source in place (regex substitution) rather than vendoring a patch/diff, so re-running install against a
newer SearXNG master could silently stop matching the regex and reintroduce Unix-only import crashes.

---

## 5. Memory (`agent/memory.py`)

**Storage:** a single SQLite file, `results/memory.sqlite`, with two content tables plus SQLite FTS5
virtual tables kept in sync via triggers:
- `turns(session, ts, role, text, interrupted)` — every turn of every session is logged (`log_turn`),
  win-inserted into `turns_fts` for BM25-ranked full-text search.
- `notes(ts, text, tags, session)` — explicit user-asked-to-remember facts, similarly FTS-indexed.
- `sessions(id, started, model, turns)` — one row per run.

**Read paths:**
- `context_for(user_text)` is called automatically before each turn (not a tool): it BM25-searches both
  notes and turns from *other* sessions, and if anything scores above `min_score` it's formatted as a
  `[memory, may be relevant: ...]` block and prepended to the user's message — this is the whole "cheap
  RAG," explicitly no embeddings/vector store.
- Four LangChain tools bound to one `Memory` instance (`make_tools`): `remember(note, tags)`,
  `recall(query)`, `search_history(query)` (past sessions only), `forget(note_id)`.
- `/memory [search <q> | notes | forget <id> | stats]` slash command in `agent/talk.py` — this is the only
  UI for memory today; there's no dedicated viewer/editor beyond ad hoc SQL or this command line.
  (`agent/brain.py` also has a much smaller, independent JSON-file fallback memory —
  `BUILTIN_TOOLS`'s `remember`/`recall` write/read `results/notes.json` directly — used only when a
  `Brain` is constructed without a `Memory` object passed in; `agent/talk.py` always passes one, so in
  practice the SQLite path is what runs.)

**Fragile / worth reconsidering:** two parallel "remember/recall" implementations exist (the SQLite-backed
one in `memory.py` and the JSON-file one baked into `brain.py`'s `BUILTIN_TOOLS`) with the same tool names;
easy to end up wired to the wrong one if `Brain` is ever instantiated differently. The BM25 keyword search
has no semantic matching — the docstring already anticipates swapping in embedding search later.

---

## 6. TTS (`tts/kokoro_stream.py`, `tts/neutts_stream.py`, `tts/neutts_worker.py`)

**Two engines, Kokoro is the active default; NeuTTS is opt-in.** Confirmed by README and `agent/talk.py`
defaults (`--tts-engine` not passed → Kokoro) and by which venv each needs:

- **Kokoro-82M** (`tts/kokoro_stream.py`): `KokoroTTS` is one resident model on CUDA via `kokoro==0.9.4` +
  `torch==2.6.0+cu124`, loaded directly in the main agent process/venv (`.venv`). `KModel`/`KPipeline`
  from the `kokoro` package; `set_voice()` swaps voice tensors without reloading. Synthesis is
  whole-segment-in/whole-waveform-out (not internally streaming) — the *pipelining* is sentence-level, done
  by this project's own `SentenceChunker` + `StreamingSpeech` classes.
- **NeuTTS-Air** (`tts/neutts_stream.py` client + `tts/neutts_worker.py` server): needs `torch>=2.8`,
  incompatible with the main venv's pinned `2.6.0+cu124`, so it runs as a **separate subprocess** in its
  own `.venv-neutts`, talking over stdin/stdout with a small length-prefixed JSON+raw-float32 protocol
  (`{"cmd": "synth_stream", ...}` → `{"event":"chunk","n":...}` + raw bytes → `{"event":"done"}`). The
  client (`NeuTTSEngine`) presents the *same* duck-typed interface as `KokoroTTS` (`.synth`,
  `.synth_stream`, `.set_voice`, `.voice`, `.vram_mib()`) so `agent/talk.py` can swap engines live with
  `/tts neutts` / `/tts kokoro` without branching logic elsewhere. A Windows Job Object
  (`_kill_with_parent`) ties the worker subprocess's lifetime to the parent so it can't leak ~5 GB of VRAM
  if the agent crashes.

**Text-ready → audio-playing interface (shared by both engines):**
`SentenceChunker.feed(piece)` accumulates streamed LLM token fragments and yields complete sentences
(splitting at `.!?…`, or at a clause break for just the *first* chunk to shave time-to-first-audio, or by
character-length fallback so a run-on sentence can't stall it indefinitely). `StreamingSpeech` runs a
generator thread that, per sentence, either calls the engine's `synth()` (Kokoro: waits for the whole
sentence's audio) or iterates `synth_stream()` if the engine exposes it (NeuTTS: pushes each sub-sentence
chunk to the speaker as it arrives — literally checked via `hasattr(self.tts, "synth_stream")`). Audio goes
into `SpeakerSink`, a `sounddevice.OutputStream` fed from a queue in its audio callback; the same played
samples are also fed back into `EchoGate.push_played()` for self-echo suppression.

**So: sentence-by-sentence, not whole-response**, with NeuTTS going one level finer (sub-sentence chunks).
Barge-in (`StreamingSpeech.interrupt()`) flushes the sink queue and stops generation; for NeuTTS this also
sends a `cancel` command across the process boundary, picked up by a dedicated stdin-reader thread within
about one internal chunk (~0.5 s) since a blocking `for line in sys.stdin` loop can't otherwise be
interrupted mid-read.

**Fragile / worth reconsidering:** the two-venv split (main vs `.venv-neutts`) exists purely because of a
transitive `torch` version conflict — an inherently fragile deployment shape (subprocess protocol,
Windows-Job-Object cleanup, registry-based env var propagation via `winenv.py`) that a rebuild should avoid
by picking one runtime story instead of coordinating two Python environments.

---

## 7. STT (`bench/common.py: load_parakeet`, `models/whisper_features.py`, mic loop in `agent/talk.py`)

**Model:** `nemo-parakeet-tdt-0.6b-v3` via the `onnx-asr` package (not raw NeMo — NeMo needs
Linux/WSL). Auto-downloaded from Hugging Face (`istupakov/parakeet-tdt-0.6b-v3-onnx`) on first use.

**Device strategy (`load_parakeet`, `bench/common.py`):** three modes — `cuda` (encoder+decoder both
GPU), `cpu`, and the default-in-practice `hybrid` (encoder on CUDA, decoder_joint moved to CPU as int8 via
`_move_decoder_to_cpu`, because the transducer decoder runs once per encoder frame with tiny tensors and is
launch-bound on GPU — CPU int8 measured faster for that specific loop). The code hard-fails
(`sys.exit`) rather than silently degrading if CUDA isn't actually active after load or after warm-up,
specifically because ONNX Runtime is known to silently fall back to CPU on some cuDNN/driver mismatches.

**Mic → text pipeline (`agent/talk.py: Talk.run` / `mic_chunks`):**
`sounddevice.InputStream` (16 kHz mono, 32 ms / 512-sample blocks) → a queue drained into fixed-size
chunks → Silero VAD scores each chunk → chunks accumulate into an utterance buffer once VAD crosses
threshold → **live partial transcripts** are produced by re-running `self.asr.recognize()` on the
growing buffer every `partial_every` seconds (a separate background thread, `partial_job`, guarded by a
non-blocking lock so it never contends with the final decode) purely for on-screen captions and to feed
the endpointer's filler-word heuristic → once the endpointer signals `"done"`/`"timeout"`, the full
utterance is decoded once more by `self.asr.recognize()` in `worker()` and handed to `Brain`.

**Fragile / worth reconsidering:** this machine's specific CUDA/cuDNN/driver pin
(`onnxruntime-gpu==1.26.0`, `nvidia-cudnn-cu12==9.1.1.17`, driver ≤551.23) is called out in the README as
fragile and machine-specific; a rebuild target (different GPU/driver) may need different pins or the
`onnxruntime` CPU path only.

---

## 8. Turn detection / barge-in (`agent/endpoint.py`, `agent/echo_gate.py`, `agent/state.py`, `agent/talk.py`)

**Not Google endpointing flags** — this is a from-scratch, two-model system:

- **End-of-turn (`Endpointer`, `agent/endpoint.py`):** Silero VAD's silence duration gates everything
  (`min_silence_ms=250` before anything is checked, `max_silence_ms=1200` hard timeout). In between, a
  **Smart Turn v3.2 ONNX model** (`models/smart-turn-v3.2-cpu.onnx`, loaded via `bench/turn_test.py:
  SmartTurn`, feature extraction in `models/whisper_features.py`) scores the last 8 s of audio every
  `check_every_ms=200`ms of silence. Three confidence tiers (added after a real bug where 0.56–0.66 scores
  cut people off mid-hedge): below 0.7 never ends; 0.7–0.9 ends *unless* the last live-partial word is a
  filler/conjunction (`FILLER_ENDINGS` set: "um", "so", "and", "that's", ...), in which case it holds;
  ≥0.9 ends regardless of the trailing word.
- **Barge-in (`agent/talk.py: Talk.run`, main mic loop):** a leaky-bucket accumulator `hot` gains a unit
  per chunk where VAD ≥ threshold and the chunk isn't classified as echo; it decays (not resets) by
  `bargein_decay` per miss so natural consonant dips in continuous speech don't erase progress. Two
  thresholds: `duck_chunks` (default corresponds to ~96 ms) ducks playback volume to `duck_gain` (a
  possible backchannel like "mhm" only gets this far and is walked back after
  `bargein_release_chunks` of real silence); `bargein_chunks` (~320 ms of net credit) actually cuts
  (`Talk.barge_in()` → `StateMachine.barge_in()` → `StreamingSpeech.interrupt()`).
- **Echo gate (`EchoGate`, `agent/echo_gate.py`):** keeps a rolling 1 s, 16 kHz copy of what the speaker
  actually played (resampled from 24 kHz via `SpeakerSink.on_play` → `push_played`), and for each mic
  chunk while the agent is talking, computes a normalized cross-correlation against that reference at
  every plausible delay (≤350 ms) plus a level check (mic must be `level_ratio`× louder than the matched
  echo segment to count as real speech). This is echo *detection for gating*, not cancellation — it never
  touches the audio Parakeet sees.
- **State machine (`agent/state.py`):** explicit `LISTENING / USER_SPEAKING / THINKING / SPEAKING /
  INTERRUPTED` enum with logged transitions; barge-in is only legal from `THINKING`/`SPEAKING`.

**Fragile / worth reconsidering:** the three-tier filler-word heuristic and the leaky-bucket barge-in
constants (`bargein-chunks`, `bargein-decay`, `bargein-release-chunks`, `duck_chunks`, thresholds) are all
hand-tuned magic numbers arrived at through live-session debugging (documented in the README's "Fixes from
the 2026-09-06 live session" section) — they will very likely need re-tuning for different mics/rooms/users
and have no adaptive/calibration mechanism.

---

## 9. Agent loop / tools (`agent/brain.py`, `agent/tools_web.py`, `agent/memory.py`, `agent/mcp.json`,
`agent/mcp_servers/`, `agent/commands.py`)

**Framework:** LangGraph via LangChain's `create_agent(model, tools=..., system_prompt=...)`
(`Brain._rebuild`), not a hand-rolled tool loop and not raw MCP-only — MCP is one *source* of tools among
several. The graph runs on a dedicated background asyncio event loop (`Brain.__init__` spins a thread
running `loop.run_forever()`), so async MCP tool calls and the agent's own sync tools coexist; the
synchronous voice loop consumes deltas via a thread-safe `queue.Queue` bridged from `agent.astream(...)`
(see `_stream_once`).

**Current tools, by source:**
- **builtin** — `current_time`, and either `remember`/`recall` (JSON-file, `brain.py`'s own, only if no
  `Memory` is passed in) or the four SQLite-backed tools from `memory.py: make_tools()` (`remember`,
  `recall`, `search_history`, `forget`) — `agent/talk.py` always supplies a `Memory`, so these are what's
  live in practice.
- **web** (only when `/search on`) — `web_search`, `read_page`, `research` from `tools_web.py`.
- **mcp:<name>** — whatever `load_mcp_tools()` (langchain-mcp-adapters) returns for each connected server
  in `agent/mcp.json`. Ships with `demo` (stdio, `agent/mcp_servers/demo_server.py`: `calculate`,
  `convert_units`, `roll_dice`, built with `mcp.server.fastmcp.FastMCP`) enabled by default, plus disabled
  examples for the official `filesystem` and `fetch` servers (stdio via `npx`/`uvx`).

**Registering a new tool:**
- A **builtin** tool: write a `@tool`-decorated function (LangChain's `langchain_core.tools.tool`) and add
  it to `BUILTIN_TOOLS` in `brain.py`, or extend `memory.py: make_tools()` if it needs the `Memory` object.
- A **web-style** tool: add a `@tool` function to `tools_web.py` and append it to `WEB_TOOLS`.
- An **MCP** tool: add a server entry to `agent/mcp.json` (`stdio` with `command`/`args`, or
  `streamable_http` with `url`), set `"enabled": true` for autoconnect or use `/mcp on <name>` at runtime;
  no agent code changes needed — `Brain.mcp_on()` connects, calls `load_mcp_tools()`, and rebuilds the
  graph. `Brain.tools` / `Brain.tool_sources()` are what `/tools` in `commands.py` reads to show origin.
- MCP sessions are long-lived (one process per server, reused across calls — reconnecting per call would
  cost ~1 s each). Turns are capped at `max_steps=12` graph steps (~5 tool round-trips) and abandoned after
  `stall_s=8` (or `tool_stall_s=25` while a tool is in flight) seconds of model silence.

**Slash commands (`agent/commands.py`)** are a separate, much simpler router (`CommandRouter`, decorator-
registered `@r.add(name, usage, help)` handlers in `agent/talk.py`) — this is the terminal-input layer
(`/model`, `/voice`, `/search`, `/mcp`, `/memory`, `/mute`, ...), distinct from and unrelated to the agent's
own LangGraph tool-calling.

**Fragile / worth reconsidering:** builtin `remember`/`recall` exists in two places with different storage
(see §5); a new contributor adding a builtin tool needs to know which `BUILTIN_TOOLS` vs `make_tools()`
path is actually live. MCP server config trusts `agent/mcp.json` verbatim (arbitrary `command`/`args`) —
fine for a local single-user tool but would need sandboxing/allow-listing before letting this run
untrusted config.

---

## 10. Dependencies — what's Python-specific vs portable

From `requirements.txt` (Python 3.11, Windows, CUDA 12.4 pinned stack) and the code read above:

**Has no direct Node/TypeScript equivalent — needs a different approach entirely:**
- `onnx-asr` + the Parakeet ONNX models + the CUDA/cuDNN pinning dance in `bench/common.py`
  (`load_parakeet`, `_move_decoder_to_cpu`, provider fallback detection). `onnxruntime-node` exists but the
  hybrid encoder/decoder placement logic and the specific NeMo/Parakeet loader (`onnx_asr.load_model`) are
  Python-package-specific; a TS rebuild would need either a different local STT engine with a Node-native
  binding, or to keep STT as an out-of-process Python service the Electron app talks to.
- `silero-vad` (Python package) — Silero VAD models are ONNX-exportable and have been used from
  Node/onnxruntime-node in other projects, but this project uses the Python package's high-level API
  directly (`load_silero_vad()`, `vad(tensor, sr)`), not the raw ONNX graph.
- `kokoro` (PyTorch `KModel`/`KPipeline`) and NeuTTS-Air's llama.cpp-backbone + torch-codec stack — both
  are Python/PyTorch-native; no maintained Node bindings for either at this pinned version. NeuTTS's
  subprocess/venv split is itself a Python-runtime workaround, not architecture worth porting.
- `models/whisper_features.py` (Smart Turn's feature extraction) and the Smart Turn ONNX model — same
  onnxruntime-node caveat as Parakeet.
- `sounddevice`/`soundfile` for mic capture and audio output — Electron/Node has its own native audio
  I/O story (e.g. `node-audiorecorder`, Web Audio API in the renderer via `getUserMedia`) that would
  replace this outright rather than port it.
- `tts/winenv.py`'s HKCU registry read for env vars — a Windows-Python-specific workaround for stale
  terminal environments; irrelevant in an Electron app, which can just read/write its own config file or
  use `app.getPath()`/electron-store.
- `search/install_searxng.py`'s NTFS-illegal-filename skipping and Unix-import patching — one-time,
  Python/SearXNG-specific install mechanics; if SearXNG is kept as an external local service at all, this
  installer logic doesn't need porting (Electron could just document/ship the same Python installer, or
  drop SearXNG for a different search API).

**Portable as design/logic, even though the current implementation is Python:**
- The **overall pipeline shape** (VAD → endpoint detection → ASR → agent → sentence-chunked TTS →
  speaker, with a barge-in state machine) — this is the architecture to keep, language-independent.
- `agent/state.py`'s explicit state machine (`LISTENING/USER_SPEAKING/THINKING/SPEAKING/INTERRUPTED` with
  logged transitions) — trivially portable to TS as-is.
- `agent/endpoint.py`'s three-tier confidence logic and `tts/kokoro_stream.py`'s `SentenceChunker`
  regex-based sentence-boundary splitting — pure text/logic algorithms, portable verbatim.
- `agent/echo_gate.py`'s cross-correlation approach — the math (normalized cross-correlation over a
  rolling reference buffer) is portable to any language with array support (e.g. via a small WASM/native
  DSP lib or just JS typed arrays).
- The **provider abstraction shape**: a single `stream_chat`-like interface over multiple LLM SDKs, a
  key-rotation pool for rate-limited providers, model aliasing — LangChain.js and the Anthropic/OpenAI/
  Google Node SDKs cover the same providers, so this is a straightforward re-implementation, not a
  re-architecture.
- The **prompt** (`llm/prompts.py`) — plain text, copy verbatim.
- The **tool-calling loop shape** (LangGraph `create_agent`-equivalent, or a hand-rolled loop) and the
  **MCP integration** — `@modelcontextprotocol/sdk` (TypeScript) is the reference MCP implementation, so
  MCP support is if anything *more* natural in a Node/Electron app than the current Python
  `langchain-mcp-adapters` dependency.
- **SQLite memory** (`agent/memory.py`) — SQLite + FTS5 is available from Node (`better-sqlite3`, or
  `node:sqlite` in modern Node) with the same schema and BM25 ranking; port directly.
- **SearXNG web tools** (`agent/tools_web.py`) — just HTTP calls (SearXNG JSON API) + HTML-to-text
  extraction (Trafilatura's role could be filled by an npm readability library, e.g. `@mozilla/readability`
  or `unfluff`); fully portable, SearXNG itself stays an external local process either way.
- **Slash command router** (`agent/commands.py`) — a trivial dispatch table, portable as-is.
