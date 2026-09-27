"""List every model each configured key can actually reach, straight from the providers.

    python llm/list_models.py            # table per provider + results/llm_models.json
    python llm/list_models.py --json     # raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)  # .env wins over any stale key already in the process env
con = Console()


def groq_models(key: str) -> list[dict]:
    r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    r.raise_for_status()
    out = []
    for m in r.json()["data"]:
        out.append({"id": m["id"], "owned_by": m.get("owned_by"), "context": m.get("context_window"),
                    "max_output": m.get("max_completion_tokens"), "active": m.get("active")})
    return sorted(out, key=lambda m: m["id"])


def anthropic_models(key: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    out = []
    for m in client.models.list():
        out.append({"id": m.id, "display": getattr(m, "display_name", None),
                    "context": getattr(m, "max_input_tokens", None), "max_output": getattr(m, "max_tokens", None),
                    "created": str(getattr(m, "created_at", ""))[:10]})
    return out


def gemini_models(key: str) -> list[dict]:
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key, "pageSize": 200}, timeout=20)
    r.raise_for_status()
    out = []
    for m in r.json().get("models", []):
        methods = m.get("supportedGenerationMethods", [])
        out.append({"id": m["name"].removeprefix("models/"), "display": m.get("displayName"),
                    "context": m.get("inputTokenLimit"), "max_output": m.get("outputTokenLimit"),
                    "chat": "generateContent" in methods, "methods": ",".join(methods)})
    return sorted(out, key=lambda m: m["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result: dict[str, object] = {}

    # Groq: check every key in the pool (they may differ in what they can reach / whether they work)
    groq_keys = json.loads(os.environ.get("GROQ_KEYS", "[]")) or [{"key": os.environ.get("GROQ_API_KEY"), "label": "GROQ_API_KEY"}]
    groq_status = []
    groq_list: list[dict] = []
    for k in groq_keys:
        try:
            ms = groq_models(k["key"])
            groq_status.append({"label": k.get("label"), "ok": True, "models": len(ms)})
            if not groq_list:
                groq_list = ms
        except Exception as e:  # noqa: BLE001
            groq_status.append({"label": k.get("label"), "ok": False, "error": str(e)[:120]})
    result["groq"] = {"keys": groq_status, "models": groq_list}

    for name, fn, env in [("anthropic", anthropic_models, "ANTHROPIC_API_KEY"), ("gemini", gemini_models, "GEMINI_API_KEY")]:
        key = os.environ.get(env)
        if not key:
            result[name] = {"error": f"{env} not set"}
            continue
        try:
            result[name] = {"models": fn(key)}
        except Exception as e:  # noqa: BLE001
            result[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    out = ROOT / "results" / "llm_models.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
        return

    t = Table(title="Groq keys")
    for c in ["label", "ok", "models / error"]:
        t.add_column(c)
    for s in groq_status:
        t.add_row(str(s["label"]), "yes" if s["ok"] else "NO", str(s.get("models", s.get("error"))))
    con.print(t)
    for name in ["groq", "anthropic", "gemini"]:
        block = result[name]
        if "error" in block:
            con.print(f"[red]{name}: {block['error']}[/]")
            continue
        models = block["models"]
        if name == "gemini":
            models = [m for m in models if m["chat"]]
        t = Table(title=f"{name}: {len(models)} models this key can use")
        for c in ["id", "context", "max output", "note"]:
            t.add_column(c, justify="right" if c != "id" and c != "note" else "left")
        for m in models:
            note = m.get("owned_by") or m.get("display") or ""
            t.add_row(m["id"], str(m.get("context") or ""), str(m.get("max_output") or ""), str(note))
        con.print(t)
    con.print(f"saved -> {out}")


if __name__ == "__main__":
    main()
