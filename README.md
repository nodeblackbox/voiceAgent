# voiceAgent — Parakeet + Silero first-pass benchmark

Native Windows, no WSL. Runtime is `onnx-asr` on `onnxruntime-gpu 1.26` (CUDA 12 build; this
machine's driver 551.23 cannot run the CUDA 13 builds that ORT >= 1.27 ships).

## Setup (already done)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe "onnx-asr[hub]" silero-vad sounddevice numpy soundfile jiwer rich huggingface_hub psutil nvidia-ml-py
uv pip install --python .venv/Scripts/python.exe "onnxruntime-gpu==1.26.0" "nvidia-cudnn-cu12==9.1.1.17" "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*" "nvidia-cufft-cu12==11.2.*" "nvidia-curand-cu12==10.3.9.*" "nvidia-cuda-nvrtc-cu12==12.4.*"
```

The cuDNN pin matters: cuDNN 9.25 (what `nvidia-cudnn-cu12==9.*` resolves to) fails on driver 551.23 with
`CUDNN_BACKEND_API_FAILED` on the first Conv, and onnxruntime then *silently* re-runs everything on CPU.
`bench/common.py` disables that fallback so it is a hard error instead, and re-checks providers after warm-up.
Updating the NVIDIA driver (>= 580) would lift both this pin and the CUDA 12 pin.

Model `nemo-parakeet-tdt-0.6b-v3` is auto-downloaded from HF (`istupakov/parakeet-tdt-0.6b-v3-onnx`) into
`~/.cache/huggingface` on first run.

## Scripts (`bench/`)

| script | what it does |
|---|---|
| `list_devices.py` | print input devices; Yeti Classic = **52** (WASAPI) or **1** (MME default) |
| `bench_offline.py` | load time, per-file latency, RTFx, VRAM, WER vs Handy's transcript of the same wav. `--device hybrid` (default: CUDA encoder + int8 CPU decoder), `cuda`, or `cpu --quant int8` |
| `profile_stages.py` | splits one call into preprocessor / encoder / decoder-loop time per device |
| `turn_test.py` | Smart Turn v3.2 end-of-turn model on the recordings and on every Silero pause of the monologue |
| `vad_test.py` | Silero VAD over `audio/*.wav` (segments, per-chunk cost) or `--live` mic probability meter |
| `live_dictate.py` | mic -> 1 s pre-roll ring buffer -> Silero gate -> Parakeet one-pass -> terminal, with speech-end-to-text latency per utterance. Logs to `results/live_*.jsonl`. `--wav file` replays a recording as a simulated mic |

```powershell
.venv\Scripts\python.exe bench\bench_offline.py --repeats 5
.venv\Scripts\python.exe bench\vad_test.py --min-silence 600
.venv\Scripts\python.exe bench\vad_test.py --live --device 52
.venv\Scripts\python.exe bench\live_dictate.py --device 52 --min-silence 600 --save-audio
.venv\Scripts\python.exe bench\live_dictate.py --wav audio\handy-1785143065.wav
.venv\Scripts\python.exe bench\profile_stages.py
.venv\Scripts\python.exe bench\turn_test.py
```

Findings are in `results/REPORT.md`.

## Text-to-speech (`tts/`)

Kokoro-82M on the GPU via `kokoro==0.9.4` and `torch==2.6.0+cu124` (install torch from
`https://download.pytorch.org/whl/cu124`; the PyPI torch wheel is CPU-only on Windows).

| file | what it does |
|---|---|
| `kokoro_stream.py` | `KokoroTTS` (one resident model), `SentenceChunker` (LLM tokens -> sentences, early first clause), `SpeakerSink` (sounddevice output, 24 kHz), `StreamingSpeech` (feed text, `interrupt()` for barge-in, metrics) |
| `tts_test.py` | offline per-sentence timing, simulated-LLM streaming with time-to-first-audio and gap count, barge-in stop time, and a round trip through Parakeet |

```powershell
.venv\Scripts\python.exe tts	ts_test.py                 # silent, all four experiments
.venv\Scripts\python.exe tts	ts_test.py --play          # through the Yeti headphone jack
.venv\Scripts\python.exe tts	ts_test.py --play --out BenQ --voice bf_emma --wps 40
```

Mic selection is now by name: `--device Yeti` (default) instead of an index, because Windows renumbers
devices between sessions.

## Data

`audio/` = the five recordings Handy kept of the Yeti (16 kHz mono), `audio/handy_reference.json` =
Handy's own Parakeet-int8 transcript of each (used as the WER reference; it is not human ground truth).

## Not done / known limits

- No word boosting: onnx-asr has no context-biasing hook. That needs NeMo (Linux/WSL) or a post-hoc corrector.
- No streaming Parakeet: `parakeet-unified` / `nemotron-speech-streaming` are not in onnx-asr; sherpa-onnx
  (`pip install sherpa-onnx==1.13.7+cuda12.cudnn9 -f https://k2-fsa.github.io/sherpa/onnx/cuda.html`) is the
  Windows route if that is ever needed.

## LLM layer (`llm/`) and the agent (`agent/`)

Keys live in `.env` (git-ignored). `llm/providers.py` wraps LiteLLM: `stream_chat(model, messages, system=...)`
yields text deltas and records first-token / first-sentence timing. Groq keys are a pool (`GROQ_KEYS`) that
rotates on 429 and permanently benches a key that returns 401.

| command | what it does |
|---|---|
| `llm/list_models.py` | asks each provider what the key can reach; writes `results/llm_models.json` |
| `llm/llm_test.py --providers groq anthropic gemini --all` | streams three spoken-style probes per candidate model, measures first token, first sentence, tok/s; checks each Groq key |
| `agent/voice_agent.py` | mic -> Silero -> Parakeet -> LLM (stream) -> Kokoro (sentence-pipelined) -> speaker, with barge-in and history |
| `agent/voice_agent.py --say "..." [--no-play] [--no-asr]` | one or more turns from text; the automated test path |
| `agent/echo_agent.py` | no LLM: repeats what you said (plumbing test) |

Defaults chosen from the measurements (2026-09-02): Groq `openai/gpt-oss-20b` (first sentence ~270-330 ms; Groq no
longer serves Llama), Anthropic `claude-opus-5` (Haiku 4.5 is ~650 ms if latency matters more than quality),
Gemini `gemini-3.5-flash-lite` (the 3.5/3.7 Flash models think first, 2-4 s). Groq key `key1_tokyo1` is invalid.

```powershell
.venv\Scripts\python.exe agent\voice_agent.py                                   # talk to it
.venv\Scripts\python.exe agent\voice_agent.py --model groq/qwen/qwen3.8-27b
.venv\Scripts\python.exe agent\voice_agent.py --model anthropic/claude-haiku-4-5
```

## The conversational agent with barge-in (`agent/talk.py`)

```
mic ─ Silero ─┬─ EchoGate ─ StateMachine ─ barge-in
              ├─ live partials (Parakeet re-decodes the growing utterance every 0.6 s, for the UI)
              └─ utterance ─ Parakeet ─ Brain (LangGraph create_agent + tools, streaming) ─ Kokoro ─ speaker
```

| file | role |
|---|---|
| `agent/talk.py` | the loop, the Rich terminal UI wiring, `--say` and `--sim` test modes |
| `agent/brain.py` | LangGraph agent (`langchain.agents.create_agent`) with tools `current_time`, `remember`, `recall`; owns history so an interrupted reply is stored as what was actually heard plus a bracketed note; incomplete tool calls are dropped |
| `agent/state.py` | LISTENING / USER_SPEAKING / THINKING / SPEAKING / INTERRUPTED with explicit transitions, all logged |
| `agent/echo_gate.py` | stops self-interruption: cross-correlates each mic chunk with the last second the speaker played; echo chunks never count toward barge-in |
| `agent/ui.py` | Rich Live screen: state, your live partial, streamed reply, tool calls, interruption markers, latency badges. `--plain` prints one line per event |

```powershell
.venv\Scripts\python.exe agent\talk.py                                   # Claude Haiku 4.5, Yeti in/out
.venv\Scripts\python.exe agent\talk.py --model groq:openai/gpt-oss-20b   # faster first sentence
.venv\Scripts\python.exe agent\talk.py --plain --say "what time is it" "remember that I like the Bella voice"
.venv\Scripts\python.exe agent\talk.py --plain --no-play --sim audio\handy-1787878656.wav --sim-bargein audio\handy-1787881187.wav --duration 60
```

Barge-in rule: 4 consecutive 32 ms chunks with VAD > 0.75 that the echo gate does not attribute to the speaker
(`--bargein-chunks`, `--bargein-threshold`, `--echo-corr`). On interrupt: Kokoro's queue is flushed (stops within one
21 ms block), the LLM stream is abandoned, the words that reached the speaker are computed from played samples, and
the model is told what was cut so "go on" continues from there.

## Web search: SearXNG finds, Trafilatura reads

No Docker on this machine, so SearXNG runs from source in its own venv (`.venv-searxng`, source in
`search/searxng-src`, one Unix-only import patched). `search/settings.yml` turns the JSON API on
(`search.formats: [html, json]`), disables the bot limiter for localhost, and disables engines that
captcha from this IP (DuckDuckGo, Qwant, Bing). Google, Brave, Startpage, Wikipedia carry general search.

```powershell
.venv\Scripts\python.exe search\run_searxng.py           # start (the agent also auto-starts it with /search on)
.venv\Scripts\python.exe search\run_searxng.py --check   # up/down
```

Tools in `agent/tools_web.py`: `web_search(query, time_range, category)` (SearXNG JSON, top 6 with snippets),
`read_page(url)` (Trafilatura, boilerplate stripped, browser UA, trimmed to 6k chars), `research(question, pages)`
(search then read the top pages in parallel). Measured: search 0.7 to 4 s depending on engine warm-up, page read 1 to 6 s.

## Slash commands, MCP, and the agent loop

Type while the agent runs (the input line is at the bottom of the screen); anything without a slash is a typed turn.

| command | effect |
|---|---|
| `/model groq:openai/gpt-oss-20b` | hot-swap the model (Anthropic, Groq, Gemini via LangChain `provider:model`) |
| `/voice af_bella` | switch Kokoro voice without reloading the model |
| `/tools` | list tools and where each comes from (builtin, web, mcp:name) |
| `/search on|off` | give or take away the web tools; `on` starts SearXNG if needed |
| `/mcp`, `/mcp on demo`, `/mcp off demo`, `/mcp reload` | servers from `agent/mcp.json` (stdio or streamable_http), attach/detach at runtime |
| `/say ...`, `/stop`, `/mute`, `/unmute`, `/history`, `/clear`, `/status`, `/help`, `/quit` | |

`agent/mcp.json` ships with a local demo server (`agent/mcp_servers/demo_server.py`: calculator, unit conversion, dice)
enabled, plus the official filesystem and fetch servers disabled as examples. MCP sessions are persistent (one process
per server); each tool round-trip is a few hundred ms. Turns are capped at 12 graph steps (about 5 tool round-trips)
and abandoned if the model goes quiet for 40 s, so a looping model cannot hang the voice loop.

### Groq rate limits

Each Groq key on this account allows 8,000 tokens per minute per model. With nine tools, the prompt and history, one
turn costs ~1,300 tokens, so a lively conversation hits the cap in a minute. The agent sets the Groq SDK to zero
retries and, on a 429, benches that key for 60 s and switches to the next working key in `GROQ_KEYS` (four working
keys = ~32k tokens/min). Tool results are trimmed to 700 chars in history after each turn so a read web page does
not sit in every later request. Verified: the switch fires mid-conversation and the next reply arrives in ~0.5 s.

## Conversational feel: endpointing, backchannels, fillers, fallbacks, memory

Added after the first review, all verified with the simulated mic (`results/t_*.log`):

| feature | what it does | evidence |
|---|---|---|
| **Smart Turn in the loop** (`agent/endpoint.py`) | after 250 ms of silence, Smart Turn v3.2 scores the utterance every 200 ms; "finished" ends the turn, otherwise it waits up to 1.2 s | 20 s monologue with 0.3-0.6 s pauses stayed one turn; the same audio with a 300 ms hangover became 5+ turns, one of them "Mm so" |
| **Two-stage barge-in** | 96 ms of speech ducks the voice to 25%; 320 ms of sustained speech cuts it; shorter bursts restore the volume and count as backchannels | 0.3 s burst: "backchannel ignored", no cut; 2.8 s utterance: cut, "heard up to" logged |
| **Filler while tools run** | if a tool has left the speaker silent for 1.5 s, one short phrase ("Still looking.") is spoken | three-page research call: filler spoken, answer followed |
| **Stall watchdog** | 8 s without model output abandons the turn (25 s while a tool is running) | written, not triggered |
| **Overload fallback** | a 529/503 retries once, then that turn runs on the fast Groq model, then back to your model | Anthropic 529 seen live; fallback path exercised by code review only |
| **Model aliases** | `/model fast` (Groq gpt-oss-20b), `smart` (Haiku 4.5), `opus`, `sonnet`, `gemini`, `qwen` | |
| **SQLite memory** (`agent/memory.py`) | every turn logged; notes; FTS5 full-text search = the cheap RAG; relevant hits attached to the user message as a `[memory ...]` block; tools remember / recall / search_history / forget; `/memory` command | new session answered "which device is my Yeti on?" from a note saved in the previous session, no tool call |

`results/memory.sqlite` is the whole memory; delete it to start fresh. Swap `Memory.search_*` for an embedding search
when the keyword RAG stops being enough.

### Test order when you talk to it (from the review)

1. Plain conversation, four turns, no tools. Pause mid-sentence on purpose: it should wait.
2. Interrupt it mid-sentence, then say "go on".
3. Say "mhm" and "yeah" while it talks: it should dip in volume and keep going.
4. "What time is it" (tool), then something current (search), then "roll two dice" (MCP).
5. `/help`, `/model fast`, `/voice af_bella`, `/search off`, `/memory notes`.

## The screen: Textual UI with a real editor (`agent/tui.py`)

`agent/talk.py` now opens a Textual app by default on a terminal (`--plain` for the line log, `--no-tui` for the old
Rich screen). Conversation blocks scroll above; a multi-line editor sits at the bottom.

| key | action |
|---|---|
| Enter | send (inside an unclosed ``` fence it inserts a newline instead) |
| Shift+Enter, Ctrl+J, Alt+Enter | newline |
| Ctrl+↑ / Ctrl+↓ | previous / next thing you sent |
| Ctrl+L | clear the editor |
| Ctrl+Q / Ctrl+C | quit |

Paste anything: bracketed paste keeps every line and nothing is sent until Enter. A paste (multi-line or > 400 chars)
becomes a purple "pasted N lines" block; the model gets the full text fenced and is asked what it is. Then use presets
on it, alone or with the text under the command:

`/explain` · `/review` · `/next` · `/summarize` · `/fix` · `/why` (optionally followed by extra words)

Everything else typed is a normal turn; slash commands work as before. Replies are spoken as well as shown.
`--no-mic` runs it as a typed chat that still talks back. The headless test `agent/tui_test.py` pastes a snippet,
runs `/review`, and checks Shift+Enter; it passes.

## Mic mute (button, F2, or "/mic") + wake word

Stops the mic from turning into turns — nothing you say reaches the model, the log, or memory while muted.
It does not touch the agent's own voice: if it's mid-reply when you mute, the reply finishes normally, only
future listening stops. Un-mute with the button (top right of the Textual screen), the F2 key from anywhere,
`/mic on`, or by saying the wake word — by default the agent's name, and "hey <name>" (e.g. "Yeti" / "hey Yeti"),
override with `--wake-word`.

Honest limit: a spoken wake word can't work with literally zero listening — something has to keep checking for
that one phrase. What actually stops while muted is everything downstream: each utterance is transcribed locally,
checked only for the wake word, and thrown away unheard if it doesn't match. Nothing is sent to the model, logged,
or written to memory unless you say the wake word.

```powershell
.venv\Scripts\python.exe agent	alk.py --wake-word lucy "hey lucy"    # custom wake phrase
```

Tests: `agent/tui_mic_test.py` drives the real button click and F2 key in a headless Textual session.
`agent/mic_mute_sim_test.py` is end-to-end with real audio (two clips synthesized by Kokoro) through the real
mic loop and real Parakeet: mutes, confirms an unrelated clip stays muted and creates no turn, confirms a
"Hey Yeti" clip un-mutes it, then confirms a normal utterance becomes a real answered turn afterward.
