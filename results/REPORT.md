# Parakeet + Silero + Smart Turn: first-pass benchmark (2026-09-02)

Machine: RTX 4090 (24 GB), driver 551.23, Windows 11, Python 3.11, native (no WSL). Mic: Blue Yeti Classic.
Runtime: onnx-asr 0.12 on onnxruntime-gpu 1.26 (CUDA 12). Model: nvidia/parakeet-tdt-0.6b-v3 (ONNX export).
Test audio: the five recordings Handy kept of your voice (1.2 s to 64 s), with Handy's own transcripts as reference.

## Headline

| metric | value |
|---|---|
| Parakeet one-pass latency, 4.4 s utterance (hybrid) | 45 ms best, 79 ms median |
| Parakeet one-pass latency, 64 s monologue (hybrid) | 147 ms best, 169 ms median (RTFx 437) |
| Model load (cached) / first-call warm-up | 4.5 s / 0.47 s |
| VRAM taken by the model | ~3.6 GB |
| Silero VAD cost per 32 ms chunk (CPU) | 0.6 ms p50, 0.7 ms p95 |
| Your natural mid-sentence pauses | 0.3 to 0.6 s |
| Smart Turn v3.2 (end-of-turn) | held 8/10 mid-monologue pauses, fired at the real end, ~70 ms/call CPU |
| Live loop speech-end to text (600 ms hangover) | ~800 ms (hangover + 160 to 240 ms ASR in-loop) |

## What went wrong and was fixed

1. **CUDA 13 mismatch.** Latest onnxruntime-gpu (1.27+) is CUDA 13; driver 551.23 tops out at CUDA 12.4. Pinned ORT 1.26 + CUDA 12 pip runtimes.
2. **Silent CPU fallback.** With cuDNN 9.25 the first Conv failed (`CUDNN_BACKEND_API_FAILED`) and onnxruntime quietly rebuilt the session on CPU. Every "CUDA" number in the first run was actually CPU (RTFx 13). Fixed by pinning cuDNN 9.1.1 and CUDA 12.4 runtimes, disabling ORT's fallback so it is a hard error, and re-checking providers after warm-up.
3. **Decoder on GPU is the bottleneck.** The transducer decoder runs once per 80 ms frame with tiny tensors. On CUDA that is launch-bound (0.5 to 0.7 ms/frame). The int8 decoder on CPU runs at 0.14 to 0.24 ms/frame. Encoder on CUDA is 30 to 40 ms even for 64 s of audio. "Hybrid" = CUDA encoder + int8 CPU decoder is now the default.

## Parakeet: stage timing (best of 3)

| mode | 4.4 s clip: enc / dec / total | 64 s clip: enc / dec / total | RTFx (64 s) |
|---|---|---|---|
| hybrid (CUDA enc + int8 CPU dec) | 33 / 14 / **51 ms** | 39 / 116 / **171 ms** | 375 |
| cuda (everything on GPU) | 31 / 37 / 70 ms | 43 / 408 / 461 ms | 139 |
| cpu int8 (Handy's path) | 317 / 15 / 342 ms | 4574 / 153 / 4787 ms | 13 |

Preprocessor is 2 to 10 ms in all modes.

## Parakeet: per-file (hybrid, best of 5)

| file | audio | best | median | RTFx | WER vs Handy |
|---|---|---|---|---|---|
| handy-1785143065 (limit-order monologue) | 64.3 s | 147 ms | 169 ms | 437 | 0.07 |
| handy-1787724741 "Oh, that's very sweet." | 1.7 s | 35 ms | 50 ms | 50 | 0.00 |
| handy-1787724752 "Thank you." | 1.2 s | 37 ms | 52 ms | 32 | 0.00 |
| handy-1787878656 "testing, testing..." | 4.4 s | 45 ms | 79 ms | 99 | 0.15 |
| handy-1787881187 "Yeah, testing..." | 2.8 s | 44 ms | 55 ms | 64 | 0.00 |

Accuracy: the only differences from Handy's transcript are filler words. Handy has filler-word removal on; Parakeet keeps "um" / "uh" and adds punctuation and capitalisation. Example: `Uh testing, testing. One, two, three. Okay. This is um you know, we're testing it.` vs Handy's `testing, testing, one two, three, okay, this is you know, we're testing it.` Real word errors on these five files: none found.

## Silero VAD

Per-chunk cost 0.58 ms p50 / 0.70 ms p95 on CPU (torch jit). First speech detected 0.1 to 0.4 s into each file.

Hangover sweep on the 64 s monologue:

| hangover | segments | gaps that split it |
|---|---|---|
| 300 ms | 11 | 0.3 to 0.6 s pauses |
| 500 ms | 3 | two 0.6 s pauses |
| 800 ms | 1 | none |
| 1200 ms | 1 | none |

Your thinking pauses sit at 0.3 to 0.6 s, so a silence-only endpointer needs 600 to 800 ms of hangover, which is then paid on every turn.

## Smart Turn v3.2 (Pipecat, CPU ONNX, 8.7 MB)

Whole recordings: all five scored DONE (p 0.70 to 0.99).

Monologue cut at each Silero pause (where a silence-only endpointer would fire):

| cut at | p(complete) | verdict |
|---|---|---|
| 2.0 s | 0.006 | not done |
| 9.9 s | 0.006 | not done |
| 13.2 s | 0.933 | DONE (end of "...it didn't visualize it.") |
| 20.4 s | 0.013 | not done |
| 40.0 s | 0.009 | not done |
| 41.0 s | 0.006 | not done |
| 44.8 s | 0.012 | not done |
| 46.6 s | 0.008 | not done |
| 47.3 s | 0.023 | not done |
| 56.1 s | 0.805 | DONE (end of "...if the price passes me and stuff like that.") |
| 63.4 s (real end) | 0.904 | DONE |

Cost 68 ms p50 / 100 ms p95 per call. The two mid-monologue DONEs are at real sentence ends, which is arguably correct behaviour for a conversational agent. This means the hangover could drop to ~300 ms with Smart Turn as the gate, saving ~300 ms per turn.

## Live loop (simulated mic, real-time replay of the recordings)

Pipeline: 512-sample chunks -> Silero -> 1 s pre-roll ring buffer -> utterance -> Parakeet hybrid.

| run | utterances | speech-end to text p50 | ASR in-loop p50 |
|---|---|---|---|
| 4.4 s clip, 600 ms hangover | 1 | 766 ms | 157 ms |
| 64 s monologue, 600 ms hangover | 4 | 807 ms | 209 ms |

In-loop ASR (157 to 241 ms) is 3 to 4x slower than the same audio offline (45 to 79 ms). Likely cause: the torch VAD loop and the CPU int8 decoder contend for the GIL/cores in one process. Next fix: Silero via onnxruntime, or run ASR in a separate process.

The live mic path opened the Yeti (WASAPI, device 52, auto-converted to 16 kHz) without errors. Not yet tested with real speech; run:

    .venv\Scripts\python.exe bench\live_dictate.py --device 52 --save-audio

## Live microphone (real speech, run by you at 04:45)

Three utterances through the Yeti (WASAPI). Device index had shifted from 52 to 47 between sessions; the scripts now select the mic by name (`--device Yeti`).

| # | audio | ASR in-loop | speech end to text | text |
|---|---|---|---|---|
| 1 | 13.9 s | 219 ms | 829 ms | "Okay, um can it hear me? No, it's not hearing me…" |
| 2 | 1.1 s | 103 ms | 714 ms | "Great." |
| 3 | 7.9 s | 212 ms | 814 ms | "Uh oh shit it is, it is, it is, it is, it is, my bad…" |

The repeated "it is" in #3 is in the audio: three different decoder configurations (int8 CPU, fp32 CPU, all-GPU) produce the same repetition, so it is not a decoder loop.

## Kokoro-82M text-to-speech (GPU, `tts/kokoro_stream.py`)

Design: Kokoro renders a whole segment at once, so streaming is pipelined at sentence level. LLM tokens go through a sentence chunker; a generator thread renders sentence N+1 while sentence N plays through a sounddevice output stream. `interrupt()` flushes the queue for barge-in.

| metric | value |
|---|---|
| load (cached) / warm-up | 4 s / 1.4 s |
| VRAM | 550 MB |
| per sentence (33 to 105 chars) | 133 to 182 ms, RTF 0.03 to 0.05 |
| time to first audio, LLM at 25 words/s | 441 ms (speakers), 490 ms (silent sink) |
| gaps between sentences (underruns) | 0 of 22 s played |
| barge-in: audio after `interrupt()` | 0 ms; playback stopped within 5 ms |
| round trip: Parakeet transcribing Kokoro | WER 0.027 on 20.7 s (only "barge-in" misheard) |

First-audio breakdown: the first clause closes at 0.28 s (7 words at 25 words/s), Kokoro takes ~150 to 200 ms, then one 21 ms audio block. A faster LLM or an earlier first-clause release lowers it further.

## LLM providers through LiteLLM (streaming, spoken-style probes, 300 max tokens)

Time to first token / first complete sentence, best and worst of three probes:

| model | first token | first sentence | note |
|---|---|---|---|
| groq/openai/gpt-oss-20b | 232 to 294 ms | 269 to 328 ms | default; reasoning set to low |
| groq/qwen/qwen3.8-27b | 136 to 448 ms | 169 to 495 ms | fastest best case |
| groq/openai/gpt-oss-120b | 334 to 342 ms | 362 to 393 ms | |
| groq/groq/compound-mini | 487 to 824 ms | 508 to 874 ms | |
| anthropic/claude-haiku-4-5 | 644 to 689 ms | 821 to 1309 ms | fastest Anthropic option |
| anthropic/claude-opus-5 | 1347 to 4134 ms | 1616 to 4134 ms | thinks before answering |
| anthropic/claude-sonnet-5 | 1978 to 3865 ms | 2648 to 5118 ms | thinks before answering |
| gemini/gemini-3.5-flash-lite | 478 to 531 ms | 532 to 579 ms | default for Gemini |
| gemini/gemini-3.7-flash | 2085 to 3674 ms | one lump | thinking ate the token budget twice |
| gemini/gemini-3.5-flash | 1570 to 2077 ms | one lump | same |

Groq no longer serves the Llama models; its chat catalogue is gpt-oss-20b/120b, Qwen 3.6/3.8 27B and compound. Groq key `key1_tokyo1` returns 401; the other four work. Anthropic key reaches 11 models, Gemini key 38.

## Full agent from text (Groq gpt-oss-20b + Kokoro, three turns)

| turn | LLM first sentence | Kokoro first sentence | turn start to first audio |
|---|---|---|---|
| "Can you hear me?" | 256 ms | 208 ms | 477 ms |
| limit vs stop order | 330 ms | 302 ms | 627 ms |
| through the headphones | 346 ms | 160 ms | 532 ms |

Add the ~800 ms speech-end-to-text from the live loop and a whole spoken turn lands at roughly 1.3 to 1.5 s from when you stop talking to when the agent starts, about 0.9 s with Smart Turn gating.

## Conversational agent with barge-in (`agent/talk.py`, Claude Haiku 4.5 via LangGraph)

Simulated mic run (utterance wav, then a second wav injected 1.5 s after the agent started talking):

| event | time | detail |
|---|---|---|
| live partials while "speaking" | every 0.6 s | "Uh testing test" -> "Uh testing testing one two three." -> full sentence, all Parakeet |
| ASR of the utterance | 125 ms | |
| agent starts speaking | 1.3 s after speech end | |
| barge-in accepted | 1.8 s into the reply | heard up to "Hey, I hear you loud", 12 words dropped, Kokoro flushed |
| second utterance answered | 1.7 s after its speech end | history shows the cut, model continued cleanly |

Turn latency with Haiku 4.5: speech end to first audio 1.7 to 1.9 s = 600 ms hangover + 125 ms ASR + 800 to 1000 ms to Haiku's first sentence + 180 ms Kokoro. Groq gpt-oss-20b in the same loop would replace the 800 to 1000 with about 300.

Tool calls: Haiku says "Let me check." first; the chunker is flushed at the tool call so that phrase is spoken during the tool round-trip. `remember` / `recall` round-trip verified across turns; interrupted turn followed by "go on" continues from the cut.

Echo gate (synthetic): 23/26 echo chunks flagged at 120 ms delay and one-quarter level; 0/31 user-speech chunks and 0/26 user-over-echo chunks wrongly flagged; 0.7 ms per chunk. Real-microphone verification with open speakers still needs a human.

## Web search, MCP and commands

SearXNG runs from source on Windows (no Docker): JSON API on, limiter off, captcha-prone engines disabled. First query 4 s (engine warm-up and captcha timeouts), later queries ~0.7 s. Trafilatura read the Hugging Face model page (18k chars) in 6.5 s and needed a browser user agent for PyPI. Haiku answered "latest trafilatura on PyPI" correctly (2.2.0, 31 July 2026) but made ten tool calls doing it; the turn is now capped at 12 graph steps and the prompt tells it to stop once it has the answer.

MCP: a local demo server (calculator, units, dice) attaches from mcp.json; Haiku and Groq both call its tools. Sessions are now persistent, so a tool round-trip no longer spawns a process. All slash commands verified in text mode: /tools, /mcp, /search on|off, /model, /voice, /status, /help.

One unexplained event: a single Groq turn took 108 s inside the agent loop while Groq alone answered in under 0.5 s; it did not recur across seven later turns. A 40 s stall watchdog now abandons such a turn instead of freezing the loop.

## After the review: endpointing, backchannels, fillers, memory

| test (simulated mic) | result |
|---|---|
| 20 s of the monologue, Smart Turn on (250 ms min, 1.2 s max) | one turn; every 0.3-0.6 s pause held; ended by the 1.2 s fallback because the clip was cut mid-sentence (score 0.13) |
| same audio, Smart Turn off, 300 ms hangover | five or more turns, one of them just "Mm so" |
| 0.3 s burst over the agent | volume ducked, "backchannel ignored", no cut |
| 2.8 s utterance over the agent | cut at 320 ms of speech, "heard up to" logged, answered the new utterance |
| three-page research call | "Still looking." spoken during the 15 s of tool time |
| note saved in one session, asked in the next | answered from the memory block with no tool call, 738 ms to first sentence |
| Anthropic 529 overloaded | seen live once; now retried once, then the turn runs on Groq |

## Recommendations

- Use hybrid mode. Budget ~50 ms ASR per utterance, ~150 ms for a minute of speech.
- Hangover 600 ms without Smart Turn; ~300 ms + Smart Turn check with it.
- Keep the 1 s pre-roll; first speech is detected 100 to 400 ms after onset.
- Update the NVIDIA driver to >= 580 when convenient; it removes both the CUDA 12 and cuDNN pins.
- Word boosting is not available in onnx-asr; a post-hoc term corrector or NeMo (WSL/Linux) would be needed.
- Full-turn budget now measurable: speech end -> text ~0.8 s (0.5 s with Smart Turn) + LLM first token + ~0.45 s to first Kokoro audio.
- GPU total for the agent: Parakeet 3.6 GB + Kokoro 0.55 GB, both resident.

Files: `results/*.json`, `results/*.log`, `results/live_*.jsonl`. Scripts in `bench/`. Setup in `README.md`.
