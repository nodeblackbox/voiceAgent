# NeuTTS-Air in this agent (second TTS engine, voice cloning)

Kokoro stays the default. NeuTTS-Air (Neuphonic, 748M-param LLM backbone + NeuCodec) is the *slow but
cloning* engine: it speaks as any 3–15 s reference clip. Same streaming interface as Kokoro
(`synth` / `synth_stream` / `set_voice`), so `agent/talk.py` swaps between them live.

## Why it lives in its own venv (and how it talks to the agent)

NeuTTS needs `torch>=2.8`; the main `.venv` is pinned to `torch==2.6.0+cu124` (last cu124 wheel) for
Parakeet/Kokoro. So NeuTTS runs in **`.venv-neutts`** as a subprocess — `tts/neutts_worker.py` — and the
main venv talks to it through `tts/neutts_stream.py` (`NeuTTSEngine`) over stdin/stdout:

    one JSON line per event, a "chunk" event immediately followed by n*4 bytes of float32 audio
    commands: synth_stream / set_voice / cancel / quit

`cancel` is read on a background thread inside the worker, so barge-in stops generation within one
internal chunk (~0.4–0.5 s), same as Kokoro in-process. **stdout belongs to the protocol**: the worker
re-points fd 1 at stderr at startup because `neutts` itself `print()`s ("Loading backbone…",
"Using seed N" on every call) and that used to desynchronise the pipe. **stderr is drained on a thread**
in the client: the codec's weight-loading progress bar alone overflows a 64 KB Windows pipe, and an
unread stderr pipe deadlocked `/tts neutts` silently (worker blocked writing, client blocked reading).

## Shutdown: the worker never outlives the agent

A Windows child does not die with its parent, and an orphaned worker sits on ~5 GB of VRAM. Four
layers, each verified 2026-09-10:

| how the agent ends | what stops the worker | measured |
|---|---|---|
| normal exit / `/quit` / Ctrl+Q | `Talk.close_tts()` → `NeuTTSEngine.close()`: `cancel` + `quit`, 5 s grace, then kill | worker gone 1.8 s after a mid-sentence `close()`; real agent: no process left, VRAM back to baseline |
| exit without `close()` (a crash that unwinds) | `atexit` runs `close()` | gone ≤ 300 ms |
| parent dies hard (crash, Task Manager, `taskkill`, `os._exit`) | Job Object with `KILL_ON_JOB_CLOSE` (pywin32) — Windows kills the worker when our handle vanishes | gone ≤ 300 ms |
| stdin EOF for any other reason | worker sets its own cancel flag, then quits; frees the llama.cpp backbone explicitly (no destructor noise) | |

The worker is also spawned in its own process group, so a console Ctrl+C in plain mode reaches only
the agent, which then shuts the worker down in order. Test: `tts/neutts_worker_test.py` (protocol) and
the shutdown checks described in `results/REPORT.md`.

## `.venv-neutts` pins (proven on this machine — RTX 4090, driver 551.23)

| package | version | note |
|---|---|---|
| torch / torchaudio | 2.11.0+cu126 | CUDA torch so the **codec runs on the GPU** (2x faster than CPU codec, see below); cu126 runs on this driver |
| llama-cpp-python | **0.3.4** (abetlen cu124 wheel) | 0.3.35's wheel dies with `0xC000001D` (illegal instruction) on this Ryzen — 0.3.4 is the build that runs |
| neutts | 1.4.1 | |
| neucodec | 0.0.6 | |
| torchao / torchtune | 0.14.1 / 0.6.1 | `tts/neutts_compat.py` shims `torchao.dtypes.nf4tensor` that torchtune imports but torchao dropped |
| transformers | 5.1.0 | |

Rebuild from scratch:

    uv venv --python 3.11 .venv-neutts
    uv pip install --python .venv-neutts/Scripts/python.exe llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    uv pip install --python .venv-neutts/Scripts/python.exe "neutts[llama]==1.4.1"
    uv pip install --python .venv-neutts/Scripts/python.exe --no-deps --reinstall torch==2.11.0+cu126 torchaudio==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python .venv-neutts/Scripts/python.exe --no-deps --reinstall torchao==0.14.1

Any other venv with these installed works too: `--neutts-python <python.exe>` or `$NEUTTS_PYTHON`.

## Weights (gated on Hugging Face)

`neuphonic/neutts-air-q8-gguf` and `neuphonic/neucodec` are gated. They are already in this machine's
shared HF cache (`HF_HOME=D:\hf-cache\huggingface`, set as a Windows User env var); a fully cached file is
served even without a token. If the cache is ever wiped: accept the terms on huggingface.co while logged
in, then `hf auth login` (or set `HF_TOKEN` at User scope — `tts/winenv.py` reads User env vars straight
from the registry, so a stale terminal doesn't matter).

## Voices

`tts/neutts_samples/<name>.wav` + `<name>.txt` (exact transcript). Bundled: dave, jo, paul, emily, greta,
juliette, mateo, sophie, steven. The encoded reference is cached next to the wav as `<name>.pt` on first use.
Your own: any 3–15 s clean clip + transcript — `/clone path.wav|the exact words spoken` in the agent.

## Using it

    python agent/talk.py --tts-engine neutts --neutts-voice paul     # start on NeuTTS
    /tts neutts        /tts kokoro          # switch live (each engine stays resident once loaded)
    /voice paul        /clone my.wav|text   # NeuTTS voices
    python tts/tts_test.py --engine neutts --voice paul --neutts-seed 1   # the 4 benchmark experiments

## Measured (2026-09-09, sequential runs, same 4 sentences, Q8 backbone on llama.cpp CUDA)

| codec | first chunk (median) | RTF (median) | load |
|---|---:|---:|---:|
| **CUDA** (torch cu126, `.venv-neutts` as pinned above) | **223 ms** | **0.34** | ~24 s |
| CPU (torch cpu, the venv's original state) | 383 ms | 0.64 | ~22 s |

Per-sentence, codec on CUDA: 1.2–2.0 s of generation for 3.3–6.2 s of audio, 7–13 streamed chunks each;
the first chunk lands before a Kokoro sentence would have finished rendering, so the sentence pipeline
in `kokoro_stream.StreamingSpeech` keeps up without gaps. Warm-up after load: ~1 s.

Kokoro on the same box: ~180 ms to first sentence, RTF ~0.014 — NeuTTS is the "cloning costs ~25x
compute" comparison point, not a Kokoro replacement. Full head-to-head: `results/REPORT.md`.
