"""LLM access for the voice agent through LiteLLM (one interface, provider chosen by config).

Provider/model strings follow LiteLLM's convention:
    groq/openai/gpt-oss-20b     anthropic/claude-opus-5     gemini/gemini-3.5-flash-lite

Keys come from the project .env (never hard-code them here). GROQ_KEYS is a JSON pool of keys that
rotates on a 429 so daily caps on one key don't stop the agent.

Everything streams. `stream_chat()` yields text deltas and fills a StreamStats object with the timing
that matters for speech: first token, first complete sentence, tokens per second.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import litellm  # noqa: E402  (after dotenv so provider env vars are visible)

litellm.suppress_debug_info = True
litellm.drop_params = True
import logging  # noqa: E402

logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # hide the per-call deprecation chatter  # silently drop params a provider doesn't support (e.g. stream_options on anthropic)

# Verified against each key with llm/list_models.py on 2026-09-02 (Groq dropped the Llama models).
DEFAULT_MODELS = {
    "groq": "groq/openai/gpt-oss-20b",
    "anthropic": "anthropic/claude-opus-5",
    "gemini": "gemini/gemini-3.5-flash-lite",   # 3.7-flash / 3.5-flash think first: 2-4 s to first word
}
CANDIDATES = {
    "groq": ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b", "groq/qwen/qwen3.8-27b", "groq/groq/compound-mini"],
    "anthropic": ["anthropic/claude-opus-5", "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5"],
    "gemini": ["gemini/gemini-3.7-flash", "gemini/gemini-3.5-flash-lite", "gemini/gemini-3.5-flash"],
}
# provider-specific knobs that keep spoken replies snappy
def provider_extra(model: str) -> dict:
    if model.startswith("groq/openai/gpt-oss") or model.startswith("groq/qwen"):
        return {"reasoning_effort": "low"}   # these are reasoning models; keep the thinking short for voice
    if model.startswith("gemini/gemini-3"):
        return {"reasoning_effort": "low"}   # litellm maps this to a small thinking budget on Gemini 3.x
    return {}


# --------------------------------------------------------------------------- key pool
@dataclass
class GroqKey:
    key: str
    label: str = ""
    proxy: str | None = None
    cooldown_until: float = 0.0


class GroqKeyPool:
    """Round-robin over GROQ_KEYS; a key that returns 429 is benched for `cooldown_s`."""

    def __init__(self, cooldown_s: float = 60.0):
        raw = os.environ.get("GROQ_KEYS")
        keys: list[GroqKey] = []
        if raw:
            for i, k in enumerate(json.loads(raw)):
                keys.append(GroqKey(k["key"], k.get("label", f"key{i}"), k.get("proxy")))
        elif os.environ.get("GROQ_API_KEY"):
            keys.append(GroqKey(os.environ["GROQ_API_KEY"], "GROQ_API_KEY"))
        if not keys:
            raise SystemExit("no Groq keys: set GROQ_KEYS or GROQ_API_KEY in .env")
        self.keys = keys
        self.cooldown_s = cooldown_s
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> GroqKey:
        with self._lock:
            now = time.time()
            for _ in range(len(self.keys)):
                k = self.keys[self._i % len(self.keys)]
                self._i += 1
                if k.cooldown_until <= now:
                    return k
            # everything is cooling down: return the one that frees up soonest
            return min(self.keys, key=lambda k: k.cooldown_until)

    def bench(self, k: GroqKey, forever: bool = False) -> None:
        k.cooldown_until = float("inf") if forever else time.time() + self.cooldown_s

    def live(self) -> list[GroqKey]:
        return [k for k in self.keys if k.cooldown_until != float("inf")]

    def first_working(self) -> GroqKey | None:
        """Probe keys (cheap GET /models) until one answers 200; dead keys are benched for good."""
        import httpx

        now = time.time()
        for k in sorted(self.live(), key=lambda k: k.cooldown_until):
            if k.cooldown_until > now:
                continue  # cooling down after a 429; try the others first
            try:
                r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {k.key}"}, timeout=8)
                if r.status_code == 200:
                    return k
                if r.status_code in (401, 403):
                    self.bench(k, forever=True)
            except Exception:  # noqa: BLE001
                continue
        return None

    def __len__(self) -> int:
        return len(self.keys)


_groq_pool: GroqKeyPool | None = None


def groq_pool() -> GroqKeyPool:
    global _groq_pool
    if _groq_pool is None:
        _groq_pool = GroqKeyPool()
    return _groq_pool


def api_key_for(model: str) -> tuple[str | None, GroqKey | None]:
    provider = model.split("/", 1)[0]
    if provider == "groq":
        k = groq_pool().next()
        return k.key, k
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY"), None
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY"), None
    return None, None


# --------------------------------------------------------------------------- streaming
@dataclass
class StreamStats:
    model: str = ""
    key_label: str = ""
    t_start: float = 0.0
    ttft_ms: float | None = None            # first non-empty text delta
    first_sentence_ms: float | None = None  # first . ! ? followed by space/end
    total_ms: float | None = None
    chunks: int = 0
    chars: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    text: str = ""
    error: str | None = None
    retries: int = 0

    @property
    def tokens_per_s(self) -> float | None:
        if self.completion_tokens and self.total_ms and self.ttft_ms is not None:
            gen_s = (self.total_ms - self.ttft_ms) / 1e3
            return self.completion_tokens / gen_s if gen_s > 0 else None
        return None


_SENT_END = re.compile(r"[.!?…]+[\"')\]]*(\s|$)")


def stream_chat(
    model: str,
    messages: list[dict],
    *,
    system: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.6,
    stats: StreamStats | None = None,
    max_retries: int = 3,
    **extra,
) -> Iterator[str]:
    """Yield text deltas from the model. Rotates Groq keys on 429; other errors propagate."""
    st = stats if stats is not None else StreamStats()
    st.model = model
    full = [{"role": "system", "content": system}] + messages if system else messages
    attempt = 0
    while True:
        key, groq_key = api_key_for(model)
        st.key_label = groq_key.label if groq_key else model.split("/")[0]
        st.t_start = time.perf_counter()
        try:
            resp = litellm.completion(
                model=model, messages=full, stream=True, max_tokens=max_tokens,
                temperature=temperature, api_key=key, stream_options={"include_usage": True},
                **{**provider_extra(model), **extra},
            )
            for chunk in resp:
                st.chunks += 1
                usage = getattr(chunk, "usage", None)
                if usage:
                    st.prompt_tokens = getattr(usage, "prompt_tokens", None) or st.prompt_tokens
                    st.completion_tokens = getattr(usage, "completion_tokens", None) or st.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                now = (time.perf_counter() - st.t_start) * 1e3
                if st.ttft_ms is None:
                    st.ttft_ms = now
                st.text += delta
                st.chars += len(delta)
                if st.first_sentence_ms is None and _SENT_END.search(st.text):
                    st.first_sentence_ms = now
                yield delta
            st.total_ms = (time.perf_counter() - st.t_start) * 1e3
            if st.completion_tokens is None:
                st.completion_tokens = max(1, st.chars // 4)  # rough fallback when no usage chunk
            return
        except (litellm.RateLimitError, litellm.AuthenticationError, litellm.BadRequestError) as e:
            # 429 -> bench the key for a minute; 401 / "Invalid API Key" -> bench it for good. Anything else re-raises.
            invalid = isinstance(e, litellm.AuthenticationError) or "Invalid API Key" in str(e)
            if groq_key is None or (not invalid and not isinstance(e, litellm.RateLimitError)):
                st.error = f"{type(e).__name__}: {e}"
                raise
            attempt += 1
            st.retries += 1
            groq_pool().bench(groq_key, forever=invalid)
            if attempt > max_retries or not groq_pool().live():
                st.error = f"{type(e).__name__}: {e}"
                raise
            time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            st.error = f"{type(e).__name__}: {e}"
            raise


def chat(model: str, messages: list[dict], **kw) -> str:
    return "".join(stream_chat(model, messages, **kw))
