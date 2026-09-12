"""Fill in Windows *User* environment variables the current process didn't inherit.

The gotcha (it bit this project twice — a silently re-downloaded 2.4 GB Parakeet model, and a
NeuTTS worker that never saw the HF token): `setx NAME value` / `[Environment]::SetEnvironmentVariable(
..., 'User')` writes HKCU\\Environment, but an already-running terminal app (Windows Terminal, VS Code,
Devin, ...) hands every new tab/child shell the environment block it captured at its OWN startup. Only a
full restart of that app — or a reboot — re-reads the registry. Rather than make that everyone's
problem, read HKCU\\Environment directly and fill in only what's missing.

Values are never logged or printed here; they go straight into os.environ. Call this before importing
anything that reads these variables (huggingface_hub, the LLM SDKs).

    from winenv import load_user_env
    load_user_env(["HF_HOME", "HF_TOKEN"])
"""
from __future__ import annotations

import os

# Where this machine's Hugging Face cache actually lives (C: has no room). Used only as a fallback when
# HF_HOME is neither in the process env nor in the registry, and only if the directory really exists.
HF_HOME_FALLBACK = r"D:\hf-cache\huggingface"


def load_user_env(names: list[str]) -> list[str]:
    """Copies each named HKCU\\Environment value into os.environ if the process doesn't already have it.
    Returns the names that were filled in (for a one-line log, never the values)."""
    filled: list[str] = []
    if os.name != "nt":
        return filled
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in names:
                if os.environ.get(name):
                    continue
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if value:
                    os.environ[name] = str(value)
                    filled.append(name)
    except OSError:
        pass
    if not os.environ.get("HF_HOME") and "HF_HOME" in names and os.path.isdir(HF_HOME_FALLBACK):
        os.environ["HF_HOME"] = HF_HOME_FALLBACK
        filled.append("HF_HOME(fallback)")
    return filled


DEFAULT_NAMES = ["HF_HOME", "HF_TOKEN", "HF_HUB_OFFLINE", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"]
