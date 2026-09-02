"""The thinking part: a LangGraph agent (LangChain `create_agent`) with tools, streamed token by token,
cancellable mid-stream, and careful about what goes into history after a barge-in.

Model spec is LangChain's "provider:model", e.g.
    anthropic:claude-haiku-4-5      groq:openai/gpt-oss-20b      google_genai:gemini-3.5-flash-lite

History is owned here, not by a LangGraph checkpointer, on purpose: after an interrupt we must record
what was actually *heard*, not what the model generated, and drop any tool call that never completed.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm"))
from providers import groq_pool  # noqa: E402  (loads .env)

from langchain.agents import create_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

NOTES = ROOT / "results" / "notes.json"


# --------------------------------------------------------------------------- tools (small, real, useful in conversation)
@tool
def current_time() -> str:
    """Current local date and time. Use when the user asks the time, date, or day."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


@tool
def remember(note: str) -> str:
    """Save a short note the user wants remembered across conversations (a fact, a preference, a todo)."""
    notes = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else []
    notes.append({"t": datetime.now().isoformat(timespec="minutes"), "note": note})
    NOTES.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"saved ({len(notes)} notes total)"


@tool
def recall(query: str = "") -> str:
    """Look up saved notes. Pass a keyword, or an empty string for the most recent five."""
    if not NOTES.exists():
        return "no notes saved yet"
    notes = json.loads(NOTES.read_text(encoding="utf-8"))
    hits = [n for n in notes if query.lower() in n["note"].lower()] if query else notes[-5:]
    return "\n".join(f"{n['t']}: {n['note']}" for n in hits) or "nothing matches"


DEFAULT_TOOLS = [current_time, remember, recall]


# --------------------------------------------------------------------------- result of one turn
@dataclass
class TurnResult:
    text: str = ""                        # full streamed text
    tool_calls: list[str] = field(default_factory=list)
    interrupted: bool = False
    ttft_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    new_messages: list[BaseMessage] = field(default_factory=list)


def _chunk_text(chunk: BaseMessage) -> str:
    t = getattr(chunk, "text", None)   # a str subclass in langchain-core 1.x; never call it
    if isinstance(t, str) and t:
        return str(t)
    c = chunk.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


class Brain:
    def __init__(self, model_spec: str = "anthropic:claude-haiku-4-5", system_prompt: str = "",
                 tools=None, temperature: float = 0.6, max_tokens: int = 400):
        provider = model_spec.split(":", 1)[0]
        if provider == "groq" and not os.environ.get("GROQ_API_KEY_SET"):
            live = groq_pool().live()
            os.environ["GROQ_API_KEY"] = live[0].key if live else os.environ.get("GROQ_API_KEY", "")
        if provider == "google_genai" and os.environ.get("GEMINI_API_KEY"):
            os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
        self.model_spec = model_spec
        self.model = init_chat_model(model_spec, temperature=temperature, max_tokens=max_tokens)
        self.tools = tools if tools is not None else DEFAULT_TOOLS
        self.system_prompt = system_prompt
        self.agent = create_agent(self.model, tools=self.tools, system_prompt=system_prompt)
        self.history: list[BaseMessage] = []
        self.lock = threading.Lock()

    def warm(self) -> None:
        try:
            self.model.invoke([HumanMessage("Reply with the single word ok.")])
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------------------------------
    def stream_turn(self, user_text: str, *, on_text, on_tool_call=None, on_tool_result=None,
                    cancel: threading.Event | None = None) -> TurnResult:
        """Run one agent turn. Calls on_text(delta) as text streams. Stops early if `cancel` is set.
        Does NOT touch history; call commit() with what was actually spoken."""
        r = TurnResult()
        t0 = time.perf_counter()
        seen_tool_ids: set[str] = set()
        inputs = {"messages": self.history + [HumanMessage(user_text)]}
        try:
            for mode, data in self.agent.stream(inputs, stream_mode=["messages", "updates"]):
                if cancel is not None and cancel.is_set():
                    r.interrupted = True
                    break
                if mode == "messages":
                    chunk, _meta = data
                    if isinstance(chunk, AIMessageChunk):
                        for tc in getattr(chunk, "tool_call_chunks", None) or []:
                            tid = tc.get("id") or tc.get("name")
                            if tc.get("name") and tid not in seen_tool_ids:
                                seen_tool_ids.add(tid)
                                r.tool_calls.append(tc["name"])
                                if r.first_sentence_ms is None and r.text.strip():
                                    r.first_sentence_ms = (time.perf_counter() - t0) * 1e3
                                if on_tool_call:
                                    on_tool_call(tc["name"])
                        text = _chunk_text(chunk)
                        if text:
                            now = (time.perf_counter() - t0) * 1e3
                            if r.ttft_ms is None:
                                r.ttft_ms = now
                            r.text += text
                            if r.first_sentence_ms is None and any(p in r.text for p in (". ", "! ", "? ", ".\n")):
                                r.first_sentence_ms = now
                            on_text(text)
                    elif isinstance(chunk, ToolMessage) and on_tool_result:
                        on_tool_result(chunk.name, str(chunk.content)[:200])
                elif mode == "updates":
                    for _node, out in (data or {}).items():
                        if isinstance(out, dict):
                            r.new_messages.extend(m for m in out.get("messages", []) if isinstance(m, BaseMessage))
        except Exception as e:  # noqa: BLE001
            r.error = f"{type(e).__name__}: {str(e)[:300]}"
        r.total_ms = (time.perf_counter() - t0) * 1e3
        if r.first_sentence_ms is None and r.text.strip():
            r.first_sentence_ms = r.total_ms  # single sentence with no trailing space: speakable at the end
        return r

    def commit(self, user_text: str, result: TurnResult, spoken_text: str, unspoken_text: str = "") -> None:
        """Write the turn into history. After a barge-in only completed tool round-trips survive, and the
        assistant message is what the user actually heard plus a note about what was cut."""
        with self.lock:
            self.history.append(HumanMessage(user_text))
            if not result.interrupted and result.new_messages:
                self.history.extend(result.new_messages)
                return
            # interrupted: keep AI(tool_calls)+ToolMessage pairs that fully completed, in order
            done_ids = {m.tool_call_id for m in result.new_messages if isinstance(m, ToolMessage)}
            kept: list[BaseMessage] = []
            for m in result.new_messages:
                if isinstance(m, AIMessage) and m.tool_calls:
                    if all(tc["id"] in done_ids for tc in m.tool_calls):
                        kept.append(AIMessage(content="", tool_calls=m.tool_calls))
                    else:
                        break
                elif isinstance(m, ToolMessage):
                    kept.append(m)
            self.history.extend(kept)
            note = ""
            if unspoken_text.strip():
                note = f"\n[interrupted by the user here; the rest was never spoken: \"{unspoken_text.strip()[:240]}\"]"
            elif not spoken_text.strip():
                note = "[interrupted by the user before saying anything]"
            self.history.append(AIMessage(content=(spoken_text.strip() + note).strip()))

    def reset(self) -> None:
        with self.lock:
            self.history.clear()
