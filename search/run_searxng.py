"""Start the local SearXNG instance (Flask dev server, fine for one user on localhost).

    .venv\Scripts\python.exe search\run_searxng.py            # foreground, Ctrl+C to stop
    .venv\Scripts\python.exe search\run_searxng.py --check    # just report whether it's up

The agent calls `ensure_running()` from here when web search is switched on.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "search" / "searxng-src"
PY = ROOT / ".venv-searxng" / "Scripts" / "python.exe"
SETTINGS = ROOT / "search" / "settings.yml"
URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")
LOG = ROOT / "results" / "searxng.log"


def is_up(timeout: float = 1.5) -> bool:
    try:
        return httpx.get(f"{URL}/healthz", timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def spawn() -> subprocess.Popen:
    env = {**os.environ, "SEARXNG_SETTINGS_PATH": str(SETTINGS), "PYTHONUNBUFFERED": "1"}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    log = LOG.open("a", encoding="utf-8")
    return subprocess.Popen([str(PY), "-m", "searx.webapp"], cwd=str(SRC), env=env, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=flags)


def ensure_running(wait_s: float = 25) -> tuple[bool, str]:
    """Start SearXNG if it isn't answering; return (up, message)."""
    if is_up():
        return True, f"SearXNG already up at {URL}"
    if not PY.exists() or not SRC.exists():
        return False, "SearXNG not installed (see README: search/)"
    p = spawn()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < wait_s:
        if is_up():
            return True, f"SearXNG started (pid {p.pid}) at {URL} in {time.perf_counter() - t0:.1f}s"
        if p.poll() is not None:
            return False, f"SearXNG exited with code {p.returncode}; see {LOG}"
        time.sleep(0.3)
    return False, f"SearXNG did not answer within {wait_s}s; see {LOG}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        print("up" if is_up() else "down", URL)
        return
    if is_up():
        print(f"already running at {URL}")
        return
    env = {**os.environ, "SEARXNG_SETTINGS_PATH": str(SETTINGS)}
    print(f"starting SearXNG from {SRC} on {URL} (settings {SETTINGS})")
    sys.exit(subprocess.call([str(PY), "-m", "searx.webapp"], cwd=str(SRC), env=env))


if __name__ == "__main__":
    main()
