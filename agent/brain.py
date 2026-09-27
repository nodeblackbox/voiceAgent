"""The thinking part: a LangGraph agent (LangChain `create_agent`) with tools, streamed token by token,
cancellable mid-stream, hot-swappable (model, tools, search, MCP servers) and careful about what goes
into history after a barge-in.

Model spec is LangChain's "provider:model", e.g.
    anthropic:claude-haiku-4-5      groq:openai/gpt-oss-20b      google_genai:gemini-3.5-flash-lite

The graph runs on a private asyncio loop in a background thread, so MCP tools (async) and our own sync
tools coexist, and the synchronous voice loop just iterates deltas from a queue.

History is owned here, not by a LangGraph checkpointer, on purpose: after an interrupt we must record
what was actually *heard*, not what the model generated, and drop any tool call that never completed.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm"))
sys.path.insert(0, str(ROOT / "agent"))
from providers import groq_pool  # noqa: E402  (loads .env)

from langchain.agents import create_agent  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

NOTES = ROOT / "results" / "notes.json"
MCP_CONFIG = ROOT / "agent" / "mcp.json"

# Names from tools_web.SOURCE_TOOLS, duplicated rather than imported so a non-search run never pays for
# tools_web's httpx/trafilatura import — see _stream_once's on_tool_result cap below.
_WEB_SOURCE_TOOLS = {"web_search", "read_page", "research"}


# --------------------------------------------------------------------------- built-in tools
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


BUILTIN_TOOLS = [current_time, remember, recall]   # JSON-file fallback when no Memory is given
DEFAULT_TOOLS = BUILTIN_TOOLS  # kept for older imports

MODEL_ALIASES = {
    "fast": "groq:openai/gpt-oss-20b",
    "smart": "anthropic:claude-haiku-4-5",
    "opus": "anthropic:claude-opus-5",
    "sonnet": "anthropic:claude-sonnet-5",
    "gemini": "google_genai:gemini-3.5-flash-lite",
    "qwen": "groq:qwen/qwen3.8-27b",
}


# --------------------------------------------------------------------------- result of one turn
@dataclass
class TurnResult:
    text: str = ""
    tool_calls: list[str] = field(default_factory=list)
    interrupted: bool = False
    ttft_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float | None = None
    error: str | None = None
    hit_step_limit: bool = False
    new_messages: list[BaseMessage] = field(default_factory=list)


def _content_text(content) -> str:
    """ToolMessage content can be a str or a list of MCP content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _chunk_text(chunk: BaseMessage) -> str:
    t = getattr(chunk, "text", None)   # a str subclass in langchain-core 1.x; never call it
    if isinstance(t, str) and t:
        return str(t)
    c = chunk.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


# --------------------------------------------------------------------------- MCP
def load_mcp_config(path: Path = MCP_CONFIG) -> dict:
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8")).get("servers", {})
    venv_py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    for name, s in cfg.items():
        s["command"] = str(s.get("command", "")).replace("${VENV_PYTHON}", venv_py)
        s["args"] = [str(a).replace("${ROOT}", str(ROOT)) for a in s.get("args", [])]
    return cfg


class Brain:
    def __init__(self, model_spec: str = "anthropic:claude-haiku-4-5", system_prompt: str = "",
                 tools=None, temperature: float = 0.6, max_tokens: int = 400, search: bool = False,
                 max_steps: int = 12, stall_s: float = 8.0, tool_stall_s: float = 25.0):
        self.max_steps = max_steps        # LangGraph recursion limit: ~5 tool round-trips per turn
        self.stall_s = stall_s            # no model output for this long -> abandon the turn (voice can't wait)
        self.tool_stall_s = tool_stall_s  # ...unless a tool is running (page reads take seconds)
        self.system_prompt = system_prompt
        self.temperature, self.max_tokens = temperature, max_tokens
        self.history: list[BaseMessage] = []
        self.lock = threading.Lock()
        self.builtin = list(tools) if tools is not None else list(BUILTIN_TOOLS)
        self.search_on = search
        self.mcp_cfg = load_mcp_config()
        self.mcp_clients: dict[str, object] = {}
        self.mcp_tools: dict[str, list] = {}
        self.groq_key = None
        self.on_notice = None   # optional callback(str) for UI notices (key rotation, trims)
        self.fallback_spec = MODEL_ALIASES["fast"]  # used for one turn when the primary model is overloaded
        # private event loop thread for the graph + MCP
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="brain-loop").start()
        self.set_model(model_spec)

    # ----------------------------------------------------------------- configuration
    def _run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def set_model(self, model_spec: str) -> str:
        model_spec = MODEL_ALIASES.get(model_spec, model_spec)
        provider = model_spec.split(":", 1)[0]
        if provider == "groq":
            k = groq_pool().first_working()
            if k is None:
                raise RuntimeError("no working Groq key in the pool")
            os.environ["GROQ_API_KEY"] = k.key
        if provider == "google_genai" and os.environ.get("GEMINI_API_KEY"):
            os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
        self.model_spec = model_spec
        kw = {}
        if provider == "groq":
            kw["max_retries"] = 0                   # a 429 must come back to us at once: we switch key, not sleep
            self.groq_key = k
        if provider == "groq" and ("gpt-oss" in model_spec or "qwen" in model_spec):
            kw["reasoning_effort"] = "low"          # reasoning models: keep the think short for voice
        self.model = init_chat_model(model_spec, temperature=self.temperature, max_tokens=self.max_tokens, **kw)
        self._rebuild()
        return model_spec

    def set_search(self, on: bool) -> str:
        self.search_on = on
        self._rebuild()
        return "web tools on: web_search, read_page, research" if on else "web tools off"

    @property
    def tools(self) -> list:
        t = list(self.builtin)
        if self.search_on:
            from tools_web import WEB_TOOLS
            t += WEB_TOOLS
        for _name, tl in self.mcp_tools.items():
            t += tl
        return t

    def tool_sources(self) -> list[tuple[str, str]]:
        out = [(t.name, "builtin") for t in self.builtin]
        if self.search_on:
            from tools_web import WEB_TOOLS
            out += [(t.name, "web") for t in WEB_TOOLS]
        for name, tl in self.mcp_tools.items():
            out += [(t.name, f"mcp:{name}") for t in tl]
        return out

    def _rebuild(self) -> None:
        self.agent = create_agent(self.model, tools=self.tools, system_prompt=self.system_prompt)

    # MCP -------------------------------------------------------------------------------------
    def mcp_status(self) -> list[dict]:
        rows = []
        for name, s in self.mcp_cfg.items():
            rows.append({"name": name, "on": name in self.mcp_tools, "tools": [t.name for t in self.mcp_tools.get(name, [])],
                         "transport": s.get("transport", "stdio"), "desc": s.get("description", "")})
        return rows

    def mcp_on(self, name: str) -> str:
        if name not in self.mcp_cfg:
            return f"no MCP server '{name}' in mcp.json (have: {', '.join(self.mcp_cfg) or 'none'})"
        if name in self.mcp_tools:
            return f"{name} already on"
        s = self.mcp_cfg[name]
        spec = {"transport": s.get("transport", "stdio")}
        if spec["transport"] == "stdio":
            spec.update(command=s["command"], args=s.get("args", []), env={**os.environ, **s.get("env", {})})
        else:
            spec.update(url=s["url"], headers=s.get("headers", {}))

        async def connect():
            # one long-lived session per server: without it the adapter re-spawns the server on every call (~1 s)
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools

            client = MultiServerMCPClient({name: spec})
            cm = client.session(name)
            session = await cm.__aenter__()
            tools = await load_mcp_tools(session, server_name=name)
            return (client, cm), tools

        client, tools = self._run(connect(), timeout=90)
        self.mcp_clients[name], self.mcp_tools[name] = client, tools
        self._rebuild()
        return f"{name} on: " + ", ".join(t.name for t in tools)

    def mcp_off(self, name: str) -> str:
        if name not in self.mcp_tools:
            return f"{name} is not on"
        self.mcp_tools.pop(name, None)
        client = self.mcp_clients.pop(name, None)
        if client:
            _c, cm = client

            async def close():
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._run(close(), timeout=10)
            except Exception:  # noqa: BLE001
                pass
        self._rebuild()
        return f"{name} off"

    def mcp_autoconnect(self) -> list[str]:
        msgs = []
        for name, s in self.mcp_cfg.items():
            if s.get("enabled"):
                try:
                    msgs.append(self.mcp_on(name))
                except Exception as e:  # noqa: BLE001
                    msgs.append(f"{name} failed: {type(e).__name__}: {str(e)[:120]}")
        return msgs

    def warm(self) -> None:
        try:
            self.model.invoke([HumanMessage("Reply with the single word ok.")])
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- one turn
    def stream_turn(self, user_text: str, *, on_text, on_tool_call=None, on_tool_result=None,
                    cancel: threading.Event | None = None) -> TurnResult:
        """Run one agent turn on the private loop; deliver deltas synchronously via callbacks."""
        primary = self.model_spec
        for attempt in range(4):
            r = self._stream_once(user_text, on_text=on_text, on_tool_call=on_tool_call,
                                  on_tool_result=on_tool_result, cancel=cancel)
            err = (r.error or "").lower()
            transient = any(k in err for k in ("overloaded", "529", "503", "502", "timeout", "connection"))
            if r.error and transient and not r.text and attempt == 0:
                time.sleep(0.4)
                continue                                   # one quick retry on the same model
            if r.error and transient and not r.text and self.model_spec != self.fallback_spec:
                if self.on_notice:
                    self.on_notice(f"{self.model_spec} is overloaded → answering this turn with {self.fallback_spec}")
                self.set_model(self.fallback_spec)
                try:
                    r = self._stream_once(user_text, on_text=on_text, on_tool_call=on_tool_call,
                                          on_tool_result=on_tool_result, cancel=cancel)
                finally:
                    self.set_model(primary)                # back to the user's choice for the next turn
                return r
            if r.error and "rate" in err and self.model_spec.startswith("groq") and not r.text:
                pool = groq_pool()
                if self.groq_key is not None:
                    pool.bench(self.groq_key)          # 60 s cool-down for the key that hit its TPM cap
                nxt = pool.first_working()
                if nxt is None or nxt is self.groq_key:
                    break
                if self.on_notice:
                    self.on_notice(f"groq rate limit on {self.groq_key.label if self.groq_key else '?'} → switching to {nxt.label}")
                self.set_model(self.model_spec)
                continue
            return r
        return r

    def _stream_once(self, user_text: str, *, on_text, on_tool_call=None, on_tool_result=None,
                     cancel: threading.Event | None = None) -> TurnResult:
        r = TurnResult()
        q: queue.Queue = queue.Queue()
        t0 = time.perf_counter()
        inputs = {"messages": self.history + [HumanMessage(user_text)]}
        agent = self.agent

        async def produce():
            try:
                async for mode, data in agent.astream(inputs, stream_mode=["messages", "updates"],
                                                      config={"recursion_limit": self.max_steps}):
                    q.put((mode, data))
                    if cancel is not None and cancel.is_set():
                        break
            except Exception as e:  # noqa: BLE001
                if "GraphRecursionError" in type(e).__name__ or "recursion" in str(e).lower():
                    q.put(("limit", None))
                else:
                    q.put(("error", f"{type(e).__name__}: {str(e)[:300]}"))
            finally:
                q.put(None)

        fut = asyncio.run_coroutine_threadsafe(produce(), self.loop)
        seen_tool_ids: set[str] = set()
        tools_in_flight = 0
        while True:
            budget = self.tool_stall_s if tools_in_flight else self.stall_s
            try:
                item = q.get(timeout=budget)
            except queue.Empty:
                r.error = f"model stalled: no output for {budget:.0f}s" + (" (tool running)" if tools_in_flight else "")
                fut.cancel()
                break
            if item is None:
                break
            mode, data = item
            if cancel is not None and cancel.is_set():
                r.interrupted = True
                fut.cancel()
                break
            if mode == "error":
                r.error = data
            elif mode == "limit":
                r.hit_step_limit = True
                if not r.text.strip():
                    msg = "I've dug through several pages and I'm going in circles; here's what I have so far."
                    r.text += msg
                    on_text(msg)
            elif mode == "messages":
                chunk, _meta = data
                if isinstance(chunk, AIMessageChunk):
                    for tc in getattr(chunk, "tool_call_chunks", None) or []:
                        tid = tc.get("id") or tc.get("name")
                        if tc.get("name") and tid not in seen_tool_ids:
                            seen_tool_ids.add(tid)
                            tools_in_flight += 1
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
                elif isinstance(chunk, ToolMessage):
                    tools_in_flight = max(0, tools_in_flight - 1)
                    if on_tool_result:
                        # web tools get more room so their source URLs survive the cap (ui.py / tui.py
                        # pull those out to show what was actually searched/read); other tools stay tight.
                        cap = 2000 if chunk.name in _WEB_SOURCE_TOOLS else 200
                        on_tool_result(chunk.name, _content_text(chunk.content)[:cap])
            elif mode == "updates":
                for _node, out in (data or {}).items():
                    if isinstance(out, dict):
                        r.new_messages.extend(m for m in out.get("messages", []) if isinstance(m, BaseMessage))
        r.total_ms = (time.perf_counter() - t0) * 1e3
        if r.first_sentence_ms is None and r.text.strip():
            r.first_sentence_ms = r.total_ms
        return r

    def commit(self, user_text: str, result: TurnResult, spoken_text: str, unspoken_text: str = "") -> None:
        with self.lock:
            self.history.append(HumanMessage(user_text))
            if not result.interrupted and result.new_messages:
                self.history.extend(self._compact(result.new_messages))
                return
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

    TOOL_KEEP = 700  # chars of each tool result kept in history after the turn (the model already used it)

    def _compact(self, msgs: list[BaseMessage]) -> list[BaseMessage]:
        out = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                txt = _content_text(m.content)
                if len(txt) > self.TOOL_KEEP:
                    m = ToolMessage(content=txt[: self.TOOL_KEEP] + " …[trimmed]", tool_call_id=m.tool_call_id, name=m.name)
            out.append(m)
        return out

    def reset(self) -> None:
        with self.lock:
            self.history.clear()

    def transcript(self, n: int = 6) -> list[tuple[str, str]]:
        out = []
        for m in self.history[-2 * n:]:
            if isinstance(m, HumanMessage):
                out.append(("you", m.content if isinstance(m.content, str) else str(m.content)))
            elif isinstance(m, AIMessage) and (m.content and not m.tool_calls):
                out.append(("agent", m.content if isinstance(m.content, str) else str(m.content)))
        return out
