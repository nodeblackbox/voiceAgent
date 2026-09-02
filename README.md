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
