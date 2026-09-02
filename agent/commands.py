"""Slash commands typed into the terminal while the agent runs.

    /help                      list commands
    /model <spec>              switch model, e.g. /model groq:openai/gpt-oss-20b
    /voice <name>              switch Kokoro voice, e.g. /voice af_bella
    /tools                     list the tools the model currently has, and their source
    /search on|off             give or take away the web tools (SearXNG + Trafilatura)
    /mcp                       list MCP servers from mcp.json with status
    /mcp on <name> | off <name> | reload
    /say <text>                send a typed turn (no mic) and speak the reply
    /mute | /unmute            stop / resume speaking replies (text still streams)
    /stop                      interrupt the current reply
    /history [n]               show the last n exchanges
    /clear                     forget the conversation
    /status                    devices, latency of the last turn, states
    /quit

Anything typed without a leading slash is treated as /say.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable


@dataclass
class Command:
    name: str
    usage: str
    help: str
    handler: Callable[[list[str]], str]


class CommandRouter:
    def __init__(self):
        self.commands: dict[str, Command] = {}

    def add(self, name: str, usage: str, help: str):
        def deco(fn):
            self.commands[name] = Command(name, usage, help, fn)
            return fn
        return deco

    def help_text(self) -> str:
        w = max(len(c.usage) for c in self.commands.values())
        return "\n".join(f"/{c.usage:<{w}}  {c.help}" for c in self.commands.values())

    def dispatch(self, line: str) -> str | None:
        """Returns a message to show, or None if nothing to show."""
        line = line.strip()
        if not line:
            return None
        if not line.startswith("/"):
            return self.commands["say"].handler([line])
        try:
            parts = shlex.split(line[1:])
        except ValueError:
            parts = line[1:].split()
        if not parts:
            return None
        name, args = parts[0].lower(), parts[1:]
        cmd = self.commands.get(name)
        if cmd is None:
            near = [c for c in self.commands if c.startswith(name[:2])]
            return f"unknown command /{name}" + (f" (did you mean /{near[0]}?)" if near else "") + "  · /help"
        try:
            return cmd.handler(args)
        except Exception as e:  # noqa: BLE001
            return f"/{name} failed: {type(e).__name__}: {e}"
