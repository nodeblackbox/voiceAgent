"""FastNeuTTS: NeuTTS with a fast prompt builder for the llama.cpp (GGUF) backend.

Why: neutts builds the GGUF prompt as one giant string containing ~400+ `<|speech_NNNN|>` special
tokens. llama.cpp's special-token matcher is O(text_len * n_special_tokens); with 65k speech tokens in
the vocab that costs several seconds per call — and `Llama.__call__` re-tokenizes, so it's paid twice.

Fix: tokenize only the short text fragments through llama.cpp (special=False, which skips the special-
token scan entirely) and map `<|speech_N|>` to its ID arithmetically. `Llama.__call__` accepts a list of
token IDs directly, so neutts's `_infer_ggml` / `_infer_stream_ggml` work unchanged — this only overrides
`_ggml_prompt`. Produces byte-identical token sequences to the slow path (see `check_fast_prompt` below).

Not our own idea — this is the user's own optimization from a working NeuTTS project, reused as-is.
"""
from __future__ import annotations

import torch
from neutts import NeuTTS
from neutts.neutts import _normalize_text


class FastNeuTTS(NeuTTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._is_quantized_model:
            self._init_fast_prompt()

    def _init_fast_prompt(self) -> None:
        llm = self.backbone

        def special_id(tok: str) -> int:
            ids = llm.tokenize(tok.encode(), add_bos=False, special=True)
            assert len(ids) == 1, f"{tok!r} did not map to a single token: {ids}"
            return ids[0]

        self._id_text_start = special_id("<|TEXT_PROMPT_START|>")
        self._id_text_end = special_id("<|TEXT_PROMPT_END|>")
        self._id_speech_start = special_id("<|SPEECH_GENERATION_START|>")
        self._id_speech_0 = special_id("<|speech_0|>")
        # Speech token IDs must be contiguous for the arithmetic mapping to be valid.
        for n in (1, 1000, 65535):
            assert special_id(f"<|speech_{n}|>") == self._id_speech_0 + n, "speech tokens not contiguous"

    def _text_ids(self, s: str) -> list[int]:
        # special=False -> plain BPE only, no special-token scan (this is what llama.cpp does
        # internally for each raw-text fragment between special tokens).
        return self.backbone.tokenize(s.encode(), add_bos=False, special=False)

    def _ggml_prompt(self, ref_codes, ref_text: str, input_text: str, emotion: str | None = None):
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.tolist()
        speech_ids = [self._id_speech_0 + int(c) for c in ref_codes]

        if self.input_format == "phonemes":
            text = f"{self._to_phones(ref_text)} {self._to_phones(input_text)}"
            return (
                self._text_ids("user: Convert the text to speech:")
                + [self._id_text_start] + self._text_ids(text) + [self._id_text_end]
                + self._text_ids("\nassistant:")
                + [self._id_speech_start] + speech_ids
            )

        ref_text = _normalize_text(ref_text)
        input_text = _normalize_text(input_text)
        if emotion is None:
            text_ids = self._text_ids(f"{ref_text} {input_text}")
        else:
            emotion_token = f"<|{emotion.upper()}|>"
            tokens = self.backbone.tokenize(emotion_token.encode(), add_bos=False, special=True)
            if len(tokens) != 1:
                raise ValueError(f"Emotion token {emotion_token} is not in the model vocab.")
            text_ids = self._text_ids(ref_text) + tokens + self._text_ids(input_text)
        return [self._id_text_start] + text_ids + [self._id_text_end] + [self._id_speech_start] + speech_ids


def check_fast_prompt(tts: FastNeuTTS, ref_codes, ref_text: str, input_text: str) -> None:
    """One-time proof that the fast prompt equals the slow (string) prompt token-for-token."""
    import time
    t = time.perf_counter()
    fast = tts._ggml_prompt(ref_codes, ref_text, input_text)
    t_fast = time.perf_counter() - t

    t = time.perf_counter()
    slow_str = NeuTTS._ggml_prompt(tts, ref_codes, ref_text, input_text)
    slow = tts.backbone.tokenize(slow_str.encode(), special=True)  # same call Llama.__call__ makes
    t_slow = time.perf_counter() - t

    assert fast == slow, f"token mismatch: len {len(fast)} vs {len(slow)}"
    print(f"fast prompt == slow prompt ({len(fast)} tokens): fast {t_fast*1000:.1f} ms, slow {t_slow*1000:.1f} ms")
