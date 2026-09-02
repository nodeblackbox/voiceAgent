"""Streaming LLM test for the voice loop.

For every provider/model: stream three spoken-style probes and measure
  ttft            first text delta
  first sentence  first complete sentence (what Kokoro can start speaking)
  tok/s           generation speed after the first token
Then a Groq key-pool check: one tiny request per key.

    python llm/llm_test.py                       # groq default model only
    python llm/llm_test.py --providers groq anthropic gemini
    python llm/llm_test.py --models groq/llama-3.1-8b-instant groq/openai/gpt-oss-20b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm"))
from prompts import PROBES, voice_system_prompt  # noqa: E402
from providers import CANDIDATES, DEFAULT_MODELS, StreamStats, groq_pool, stream_chat  # noqa: E402

con = Console()


def run_model(model: str, probes: list[str], max_tokens: int) -> list[dict]:
    rows = []
    system = voice_system_prompt(user_name="Nasan")
    for p in probes:
        st = StreamStats()
        try:
            text = "".join(stream_chat(model, [{"role": "user", "content": p}], system=system,
                                       max_tokens=max_tokens, stats=st))
        except Exception as e:  # noqa: BLE001
            rows.append({"model": model, "probe": p, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            con.print(f"[red]{model}: {type(e).__name__}: {str(e)[:200]}[/]")
            continue
        rows.append({"model": model, "key": st.key_label, "probe": p, "ttft_ms": round(st.ttft_ms or 0),
                     "first_sentence_ms": round(st.first_sentence_ms or 0) if st.first_sentence_ms else None,
                     "total_ms": round(st.total_ms or 0), "completion_tokens": st.completion_tokens,
                     "tok_s": round(st.tokens_per_s or 0, 1), "retries": st.retries, "text": text})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="*", default=["groq"], choices=list(DEFAULT_MODELS))
    ap.add_argument("--models", nargs="*", default=[], help="explicit litellm model strings (override providers)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-pool-check", action="store_true")
    ap.add_argument("--all", action="store_true", help="test every CANDIDATES model of the chosen providers")
    args = ap.parse_args()

    models = args.models or [m for p in args.providers for m in (CANDIDATES[p] if args.all else [DEFAULT_MODELS[p]])]
    all_rows: list[dict] = []
    for m in models:
        con.print(f"\n[bold]{m}[/]")
        rows = run_model(m, PROBES, args.max_tokens)
        all_rows += rows
        t = Table()
        for c in ["ttft ms", "1st sentence ms", "total ms", "tokens", "tok/s", "key", "reply"]:
            t.add_column(c, justify="right" if "ms" in c or c in ("tokens", "tok/s") else "left")
        for r in rows:
            if "error" in r:
                t.add_row("-", "-", "-", "-", "-", "-", r["error"])
            else:
                t.add_row(str(r["ttft_ms"]), str(r["first_sentence_ms"]), str(r["total_ms"]), str(r["completion_tokens"]),
                          str(r["tok_s"]), r["key"], r["text"][:110].replace("\n", " "))
        con.print(t)

    pool = None
    if not args.no_pool_check and any(m.startswith("groq/") for m in models):
        con.print("\n[bold]Groq key pool: one tiny request per key[/]")
        pool = []
        for k in groq_pool().keys:
            st = StreamStats()
            t0 = time.perf_counter()
            try:
                import litellm

                r = litellm.completion(model="groq/openai/gpt-oss-20b", api_key=k.key, max_tokens=5,
                                       messages=[{"role": "user", "content": "Say ok."}])
                ok, note = True, r.choices[0].message.content.strip()[:20]
            except Exception as e:  # noqa: BLE001
                ok, note = False, f"{type(e).__name__}: {str(e)[:100]}"
            ms = (time.perf_counter() - t0) * 1e3
            pool.append({"label": k.label, "ok": ok, "ms": round(ms), "note": note, "proxy": bool(k.proxy)})
            con.print(f"  {k.label:14} {'ok ' if ok else 'BAD'} {ms:5.0f} ms  {note}{'  (has proxy entry, unused)' if k.proxy else ''}")

    out = ROOT / "results" / "llm_test.json"
    out.write_text(json.dumps({"rows": all_rows, "pool": pool}, indent=2, ensure_ascii=False), encoding="utf-8")
    con.print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
