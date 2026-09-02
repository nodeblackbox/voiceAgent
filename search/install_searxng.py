"""Reproducible SearXNG install for Windows without Docker.

    .venv\Scripts\python.exe search\install_searxng.py

What it does (all idempotent):
  1. downloads the current SearXNG source archive from GitHub
  2. extracts it to search/searxng-src, skipping the four files whose names are illegal on NTFS
  3. guards the Unix-only imports (pwd/grp/fcntl/...) so the app imports on Windows
  4. creates .venv-searxng (Python 3.11 via uv) and installs searxng-src/requirements.txt into it
Then `search/run_searxng.py` starts it with search/settings.yml.
"""
from __future__ import annotations

import io
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "search" / "searxng-src"
VENV = ROOT / ".venv-searxng"
ARCHIVE = "https://github.com/searxng/searxng/archive/refs/heads/master.zip"
UNIX_ONLY = ("pwd", "grp", "fcntl", "resource", "termios", "pty")


def fetch_source() -> None:
    print("downloading", ARCHIVE)
    data = httpx.get(ARCHIVE, follow_redirects=True, timeout=180).content
    z = zipfile.ZipFile(io.BytesIO(data))
    if SRC.exists():
        shutil.rmtree(SRC)
    skipped = 0
    for m in z.infolist():
        parts = m.filename.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        rel = parts[1]
        if any(ch in rel for ch in ':*?"<>|'):
            skipped += 1
            continue
        out = SRC / rel
        if m.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(m))
    print(f"extracted to {SRC} (skipped {skipped} NTFS-illegal names)")


def patch_unix_imports() -> None:
    n = 0
    pat = re.compile(r"^(import (%s))\s*$" % "|".join(UNIX_ONLY), re.M)
    for p in (SRC / "searx").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        new = pat.sub(r"try:\n    \1\nexcept ImportError:  # Windows\n    \2 = None", s)
        if new != s:
            p.write_text(new, encoding="utf-8")
            n += 1
            print("patched", p.relative_to(ROOT))
    print(f"{n} file(s) patched")


def make_venv() -> None:
    py = VENV / "Scripts" / "python.exe"
    if not py.exists():
        subprocess.check_call(["uv", "venv", "--python", "3.11", str(VENV)])
    subprocess.check_call(["uv", "pip", "install", "--python", str(py), "-r", str(SRC / "requirements.txt")])
    subprocess.check_call([str(py), "-c", "import searx; print('searx import ok')"], cwd=str(SRC))


if __name__ == "__main__":
    fetch_source()
    patch_unix_imports()
    make_venv()
    print("done. start with: .venv\\Scripts\\python.exe search\\run_searxng.py")
    sys.exit(0)
