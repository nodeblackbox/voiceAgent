"""SQLite memory for the agent: every turn is logged, notes are kept, and both are searchable with
SQLite's FTS5 full-text index (BM25 ranking). That is the "cheap RAG": no embeddings, no vector store,
one file (results/memory.sqlite), sub-millisecond queries. Swap `search()` for an embedding search later
without touching the callers.

Used three ways:
  * tools: remember(note), recall(query), search_history(query)
  * automatic context: before each turn, `context_for(user_text)` pulls the top hits from notes and
    past sessions and appends them to the user message as a bracketed memory block
  * /memory command: stats, search, forget
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "results" / "memory.sqlite"

_STOP = set("the a an and or but if then so of to in on at for with is are was were be been it its this that these those "
            "i you we they he she me my your our it's what which who how when where why do does did can could would should "
            "just like um uh okay ok yeah yes no not have has had about there here".split())


def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9'\-]{1,}", text.lower())
    return [w for w in words if w not in _STOP][:12]


class Memory:
    def __init__(self, path: Path = DB_PATH, session_id: str | None = None):
        path.parent.mkdir(exist_ok=True)
        self.path = path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._schema()

    def _schema(self) -> None:
        c = self.db
        c.executescript("""
        CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, session TEXT, ts REAL, role TEXT, text TEXT, interrupted INTEGER DEFAULT 0);
        CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(text, content='turns', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text); END;
        CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN INSERT INTO turns_fts(turns_fts, rowid, text) VALUES('delete', old.id, old.text); END;
        CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, ts REAL, text TEXT, tags TEXT DEFAULT '', session TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(text, tags, content='notes', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN INSERT INTO notes_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags); END;
        CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN INSERT INTO notes_fts(notes_fts, rowid, text, tags) VALUES('delete', old.id, old.text, old.tags); END;
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, started REAL, model TEXT, turns INTEGER DEFAULT 0);
        """)
        c.execute("INSERT OR IGNORE INTO sessions(id, started) VALUES (?, ?)", (self.session_id, time.time()))
        c.commit()

    # ---------------------------------------------------------------- writes
    def log_turn(self, role: str, text: str, interrupted: bool = False) -> None:
        if not text.strip():
            return
        with self.lock:
            self.db.execute("INSERT INTO turns(session, ts, role, text, interrupted) VALUES (?,?,?,?,?)",
                            (self.session_id, time.time(), role, text.strip(), int(interrupted)))
            self.db.execute("UPDATE sessions SET turns = turns + 1 WHERE id = ?", (self.session_id,))
            self.db.commit()

    def add_note(self, text: str, tags: str = "") -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO notes(ts, text, tags, session) VALUES (?,?,?,?)",
                                  (time.time(), text.strip(), tags, self.session_id))
            self.db.commit()
            return cur.lastrowid

    def forget(self, note_id: int) -> bool:
        with self.lock:
            n = self.db.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount
            self.db.commit()
            return n > 0

    # ---------------------------------------------------------------- reads
    @staticmethod
    def _fts_query(text: str) -> str | None:
        terms = _terms(text)
        if not terms:
            return None
        return " OR ".join(f'"{t}"' for t in terms)

    def search_notes(self, query: str, k: int = 5) -> list[dict]:
        q = self._fts_query(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT n.id, n.ts, n.text, n.tags, bm25(notes_fts) AS r FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
            "WHERE notes_fts MATCH ? ORDER BY r LIMIT ?", (q, k)).fetchall()
        return [{"id": i, "when": _when(ts), "text": t, "tags": g, "score": -r} for i, ts, t, g, r in rows]

    def search_turns(self, query: str, k: int = 5, exclude_session: str | None = None) -> list[dict]:
        q = self._fts_query(query)
        if not q:
            return []
        sql = ("SELECT t.id, t.session, t.ts, t.role, t.text, bm25(turns_fts) AS r FROM turns_fts JOIN turns t ON t.id = turns_fts.rowid "
               "WHERE turns_fts MATCH ? " + ("AND t.session != ? " if exclude_session else "") + "ORDER BY r LIMIT ?")
        args = (q, exclude_session, k) if exclude_session else (q, k)
        rows = self.db.execute(sql, args).fetchall()
        return [{"id": i, "session": s, "when": _when(ts), "role": role, "text": t, "score": -r} for i, s, ts, role, t, r in rows]

    def recent_notes(self, k: int = 5) -> list[dict]:
        rows = self.db.execute("SELECT id, ts, text, tags FROM notes ORDER BY id DESC LIMIT ?", (k,)).fetchall()
        return [{"id": i, "when": _when(ts), "text": t, "tags": g} for i, ts, t, g in rows]

    def context_for(self, user_text: str, k_notes: int = 3, k_turns: int = 3, min_score: float = 0.0) -> str:
        """The cheap-RAG block: relevant notes + lines from earlier sessions, or '' if nothing relevant."""
        notes = [n for n in self.search_notes(user_text, k_notes) if n["score"] >= min_score]
        turns = [t for t in self.search_turns(user_text, k_turns, exclude_session=self.session_id) if t["score"] >= min_score]
        if not notes and not turns:
            return ""
        lines = ["[memory, may be relevant:"]
        for n in notes:
            lines.append(f"  note ({n['when']}): {n['text'][:200]}")
        for t in turns:
            lines.append(f"  {t['role']} said ({t['when']}): {t['text'][:200]}")
        lines.append("]")
        return "\n".join(lines)

    def stats(self) -> dict:
        t = self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        n = self.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        s = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {"turns": t, "notes": n, "sessions": s, "db": str(self.path), "session": self.session_id}


def _when(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    today = datetime.now().date()
    if d.date() == today:
        return d.strftime("today %H:%M")
    return d.strftime("%d %b %H:%M")


# ---------------------------------------------------------------- LangChain tools bound to one Memory
def make_tools(mem: Memory):
    from langchain_core.tools import tool

    @tool
    def remember(note: str, tags: str = "") -> str:
        """Save something the user wants remembered across conversations: a fact, a preference, a decision, a todo.
        Keep it one sentence. Optional comma-separated tags."""
        i = mem.add_note(note, tags)
        return f"saved note #{i}"

    @tool
    def recall(query: str = "") -> str:
        """Search saved notes. Pass keywords, or an empty string for the most recent notes."""
        rows = mem.search_notes(query, 6) if query.strip() else mem.recent_notes(6)
        if not rows:
            return "no matching notes"
        return "\n".join(f"#{r['id']} ({r['when']}): {r['text']}" for r in rows)

    @tool
    def search_history(query: str) -> str:
        """Search everything said in earlier conversations (both sides) for keywords. Use when the user refers
        to something from a previous session ('what did I say about...', 'that thing we discussed')."""
        rows = mem.search_turns(query, 6)
        if not rows:
            return "nothing in past conversations matches"
        return "\n".join(f"{r['when']} {r['role']}: {r['text'][:220]}" for r in rows)

    @tool
    def forget(note_id: int) -> str:
        """Delete a saved note by its number (from recall)."""
        return f"deleted note #{note_id}" if mem.forget(note_id) else f"no note #{note_id}"

    return [remember, recall, search_history, forget]
