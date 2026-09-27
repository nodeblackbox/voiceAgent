"""Web tools for the agent: SearXNG finds, Trafilatura reads.

  web_search(query, ...)  -> ranked hits from the local SearXNG instance (JSON API)
  read_page(url, ...)     -> the readable text of one page, boilerplate stripped, trimmed for speech

Both are plain LangChain tools; `WEB_TOOLS` is what brain.py adds when search is on.
SearXNG must be running (see search/run_searxng.py); its URL comes from SEARXNG_URL (default
http://127.0.0.1:8888). Nothing here is spoken directly: the model reads the result and summarises.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_core.tools import tool

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) voiceAgent/1.0"


def searxng_ok(timeout: float = 2.0) -> tuple[bool, str]:
    try:
        r = httpx.get(f"{SEARXNG_URL}/healthz", timeout=timeout)
        if r.status_code == 200:
            return True, "up"
        return False, f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}"


def searx_search(query: str, *, categories: str = "general", language: str = "en", time_range: str = "",
                 max_results: int = 6, engines: str = "", safesearch: int = 0) -> dict:
    params = {"q": query, "format": "json", "categories": categories, "language": language,
              "safesearch": safesearch, "pageno": 1}
    if time_range in ("day", "week", "month", "year"):
        params["time_range"] = time_range
    if engines:
        params["engines"] = engines
    t0 = time.perf_counter()
    r = httpx.get(f"{SEARXNG_URL}/search", params=params, headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    data = r.json()
    hits = []
    for x in data.get("results", [])[:max_results]:
        hits.append({"title": x.get("title", "").strip(), "url": x.get("url", ""),
                     "snippet": re.sub(r"\s+", " ", x.get("content", "") or "").strip()[:300],
                     "engine": ",".join(x.get("engines", []) or [x.get("engine", "")]),
                     "date": (x.get("publishedDate") or "")[:10]})
    return {"query": query, "hits": hits, "answers": [str(a) for a in data.get("answers", [])][:3],
            "infobox": [{"title": i.get("infobox"), "content": (i.get("content") or "")[:400]} for i in data.get("infoboxes", [])][:1],
            "suggestions": data.get("suggestions", [])[:5], "ms": round((time.perf_counter() - t0) * 1e3),
            "total": data.get("number_of_results")}


_PYPI = re.compile(r"https?://pypi\.org/project/([A-Za-z0-9_.\-]+)/?")


def _pypi_json(url: str, max_chars: int) -> dict | None:
    """PyPI serves scrapers a JS challenge; its JSON API is open. Turn a project page into readable text."""
    m = _PYPI.match(url)
    if not m:
        return None
    name = m.group(1)
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"https://pypi.org/pypi/{name}/json", timeout=10, headers={"User-Agent": UA})
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # noqa: BLE001
        return {"url": url, "error": f"PyPI JSON API: {type(e).__name__}"}
    info = d["info"]
    rels = d.get("releases", {})
    latest = info["version"]
    when = ""
    files = rels.get(latest) or []
    if files:
        when = min(f.get("upload_time", "") for f in files)[:10]
    recent = sorted(((min(f.get("upload_time", "") for f in fs)[:10] if fs else "", v) for v, fs in rels.items()), reverse=True)[:6]
    nl = "\n"
    text = (f"{info['name']} latest version {latest}" + (f" released {when}" if when else "") + nl
            + f"summary: {info.get('summary') or ''}{nl}requires python: {info.get('requires_python') or '?'}{nl}"
            + "recent releases: " + ", ".join(f"{v} ({d_})" for d_, v in recent) + nl + nl
            + (info.get("description") or "")[:max_chars])
    return {"url": url, "title": f"{info['name']} on PyPI", "author": info.get("author"), "date": when,
            "chars": len(text), "truncated": False, "text": text[:max_chars], "ms": round((time.perf_counter() - t0) * 1e3)}


def fetch_readable(url: str, max_chars: int = 6000, markdown: bool = False) -> dict:
    from copy import deepcopy

    special = _pypi_json(url, max_chars)
    if special is not None:
        return special

    import trafilatura
    from trafilatura.settings import DEFAULT_CONFIG

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["DEFAULT"]["DOWNLOAD_TIMEOUT"] = "12"
    cfg["DEFAULT"]["USER_AGENTS"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/128.0 Safari/537.36")  # PyPI and others block the default trafilatura UA
    t0 = time.perf_counter()
    downloaded = trafilatura.fetch_url(url, config=cfg)
    if not downloaded:
        return {"url": url, "error": "could not download (blocked, timeout, or not HTML)"}
    meta = trafilatura.bare_extraction(downloaded, url=url, with_metadata=True, favor_precision=True,
                                       include_tables=True, config=cfg)
    text = trafilatura.extract(downloaded, url=url, output_format="markdown" if markdown else "txt",
                               include_links=False, include_tables=True, favor_precision=True, config=cfg) or ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = len(text) > max_chars
    return {"url": url, "title": getattr(meta, "title", None) if meta else None,
            "author": getattr(meta, "author", None) if meta else None,
            "date": getattr(meta, "date", None) if meta else None,
            "chars": len(text), "truncated": truncated, "text": text[:max_chars],
            "ms": round((time.perf_counter() - t0) * 1e3)}


# --------------------------------------------------------------------------- LangChain tools
@tool
def web_search(query: str, time_range: str = "", category: str = "general") -> str:
    """Search the web (via a local SearXNG metasearch) and get the top results with snippets.
    Use for anything current, factual, or outside your own knowledge. `time_range` can be day, week,
    month, year or empty. `category` is general, news, science, it, images or videos.
    Returns numbered results with title, url, snippet. Follow up with read_page(url) for detail."""
    try:
        res = searx_search(query, categories=category, time_range=time_range)
    except Exception as e:  # noqa: BLE001
        return f"search failed: {type(e).__name__}: {str(e)[:120]}. Is SearXNG running?"
    lines = []
    if res["answers"]:
        lines.append("direct answer: " + " | ".join(res["answers"]))
    if res["infobox"]:
        ib = res["infobox"][0]
        lines.append(f"infobox {ib['title']}: {ib['content']}")
    for i, h in enumerate(res["hits"], 1):
        d = f" ({h['date']})" if h["date"] else ""
        lines.append(f"{i}. {h['title']}{d}\n   {h['url']}\n   {h['snippet']}")
    if not res["hits"]:
        lines.append("no results")
    return "\n".join(lines)


@tool
def read_page(url: str, max_chars: int = 6000) -> str:
    """Fetch one web page and return its readable text with navigation, ads and boilerplate removed.
    Use after web_search when a snippet is not enough. Long pages are truncated to max_chars."""
    r = fetch_readable(url, max_chars=max_chars)
    if "error" in r:
        return f"{r['url']}: {r['error']}"
    head = f"{r['title'] or url}" + (f" — {r['author']}" if r["author"] else "") + (f" ({r['date']})" if r["date"] else "")
    return f"{head}\n\n{r['text']}" + ("\n\n[truncated]" if r["truncated"] else "")


@tool
def research(question: str, pages: int = 3) -> str:
    """Search, then read the top pages in parallel and return their text together (for questions that
    need more than snippets). Slower than web_search (a few seconds). pages: how many results to read (1-5)."""
    try:
        res = searx_search(question, max_results=max(1, min(pages, 5)))
    except Exception as e:  # noqa: BLE001
        return f"search failed: {type(e).__name__}: {str(e)[:120]}"
    urls = [h["url"] for h in res["hits"] if h["url"].startswith("http") and not h["url"].lower().endswith(".pdf")]
    if not urls:
        return "no readable results"
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        pages_out = list(ex.map(lambda u: fetch_readable(u, max_chars=2500), urls))
    parts = []
    for p in pages_out:
        if "error" in p:
            parts.append(f"## {p['url']}\n(unreadable: {p['error']})")
        else:
            parts.append(f"## {p['title'] or p['url']}\n{p['url']}\n{p['text']}")
    return "\n\n".join(parts)


WEB_TOOLS = [web_search, read_page, research]

# Tools whose result text is worth mining for source URLs, so the UI can show what it actually looked
# at instead of a truncated blob of prose (see ui.py / tui.py `tool()`).
SOURCE_TOOLS = {"web_search", "read_page", "research"}
_URL_RE = re.compile(r"https?://[^\s)\]]+")


def extract_sources(text: str, limit: int = 8) -> list[str]:
    """Pull the distinct URLs out of a web_search/research/read_page result, in the order they first
    appear, so the UI can list "sources" without re-parsing the whole tool-call plumbing."""
    seen: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;:")
        if u not in seen:
            seen.append(u)
        if len(seen) >= limit:
            break
    return seen


def sources_for(tool_name: str, result: str) -> list[str]:
    """What ui.py / tui.py call: the source URLs for a finished tool call, or [] if `tool_name` isn't
    one that reads the web (so callers fall back to their normal truncated-result display)."""
    if tool_name not in SOURCE_TOOLS:
        return []
    return extract_sources(result)
