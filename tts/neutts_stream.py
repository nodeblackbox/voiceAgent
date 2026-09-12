"""NeuTTS-Air client for the main agent venv: talks to `neutts_worker.py` running in `.venv-neutts` (a
separate venv — neutts needs torch>=2.8.0, incompatible with the torch==2.6.0+cu124 that Kokoro/Parakeet/
the agent depend on here) over a stdin/stdout pipe, and presents the exact same duck-typed interface as
`tts/kokoro_stream.py`'s `KokoroTTS`: `.synth`, `.synth_stream`, `.voice`, `.set_voice()`, `.load_s`,
`.warmup_s`, `.vram_mib()`, `.device` — so `agent/talk.py` can swap between the two with one command.

Backend: the Q8 GGUF backbone on llama.cpp/CUDA (the only NeuTTS backend that streams) with the fast
prompt builder in `fast_neutts.py`. Cloning: point it at a 3-15 s reference clip + transcript once;
`encode_reference()` caches the encoded reference next to the audio as `<name>.pt`.

The Neuphonic model repos on Hugging Face are gated — you must accept their terms on huggingface.co
while logged into your own account, then `hf auth login` (or set HF_TOKEN) before the worker can
download them. See tts/README_NEUTTS.md.

Barge-in: cancelling mid-sentence sends a `cancel` command the worker's background stdin-reader thread
picks up immediately, so generation stops within about one internal chunk (~0.5 s) — see
`neutts_worker.py`'s docstring for why a plain blocking read can't do that.
"""
from __future__ import annotations

import atexit
import collections
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


def _kill_with_parent(proc: subprocess.Popen):
    """Windows children outlive a parent that dies hard (crash, Task Manager, taskkill) — an orphaned
    worker would sit on ~5 GB of VRAM until someone finds it. A Job Object with KILL_ON_JOB_CLOSE ties the
    worker's life to this process: when our handle goes away for any reason, Windows kills the worker.
    Returns the job handle (keep it alive) or None if pywin32 isn't available."""
    if os.name != "nt":
        return None
    try:
        import win32api
        import win32job

        job = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
        win32job.AssignProcessToJobObject(job, win32api.OpenProcess(0x1F0FFF, False, proc.pid))
        return job
    except Exception:  # noqa: BLE001  (no pywin32, or already in a job that forbids nesting)
        return None

TTS_DIR = Path(__file__).resolve().parent
VENV_PYTHON = TTS_DIR.parent / ".venv-neutts" / "Scripts" / "python.exe"
SAMPLES_DIR = TTS_DIR / "neutts_samples"
DEFAULT_BACKBONE = "neuphonic/neutts-air-q8-gguf"
DEFAULT_CODEC = "neuphonic/neucodec"


def bundled_voices() -> list[str]:
    return sorted(p.stem for p in SAMPLES_DIR.glob("*.wav")) if SAMPLES_DIR.exists() else []


class NeuTTSWorkerError(RuntimeError):
    pass


class NeuTTSEngine:
    """Spawns and owns one `neutts_worker.py` subprocess. `speed` is accepted for interface parity with
    Kokoro but not applied — NeuTTS has no built-in rate control."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, codec: str = DEFAULT_CODEC, device: str = "cuda",
                voice: str = "dave", start_timeout: float = 240.0, python: str | Path | None = None,
                codec_device: str = "auto", seed: int | None = None):
        # which venv runs the worker: explicit arg > $NEUTTS_PYTHON > .venv-neutts next to this repo
        py = Path(python or os.environ.get("NEUTTS_PYTHON") or VENV_PYTHON)
        if not py.exists():
            raise SystemExit(
                f"NeuTTS venv python not found at {py}.\n"
                f"Set it up once: uv venv --python 3.11 {VENV_PYTHON.parent.parent}\n"
                f"then install neutts there — see tts/README_NEUTTS.md — or point --neutts-python / $NEUTTS_PYTHON "
                f"at an existing venv that has neutts + llama-cpp-python (CUDA) installed."
            )
        self.python = py
        self.device, self.backbone_repo, self.codec_repo = device, backbone, codec
        self._lock = threading.Lock()
        self._next_id = 0
        t0 = time.perf_counter()
        cmd = [str(py), str(TTS_DIR / "neutts_worker.py"), "--backbone", backbone, "--codec", codec,
               "--device", device, "--voice", voice, "--samples-dir", str(SAMPLES_DIR), "--codec-device", codec_device]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        # Own process group: in plain (non-TUI) mode a console Ctrl+C is delivered to every process on
        # the console; without this the worker would get a KeyboardInterrupt mid-generation instead of
        # the orderly cancel/quit we send it.
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=0, creationflags=flags)
        self._closed = False
        self._job = _kill_with_parent(self.proc)
        atexit.register(self.close)
        # Drain stderr continuously. The codec's weight-loading progress bar alone writes far more than
        # a Windows pipe buffer (64 KB); with nobody reading, the worker blocks on its stderr write while
        # we block on stdout.readline() -> a silent deadlock at "loading neutts…" (exactly the hang seen
        # on /tts neutts). Keep only a tail for error messages.
        self._stderr_tail: collections.deque[bytes] = collections.deque(maxlen=400)
        threading.Thread(target=self._drain_stderr, daemon=True, name="neutts-stderr").start()
        ready = self._read_control_line(timeout=start_timeout)
        if ready is None or ready.get("event") != "ready":
            self._raise_startup_error(ready)
        self.load_s = ready["load_s"]
        self.warmup_s = ready["warmup_s"]
        self.voice = ready["voice"]
        self.codec_device = ready.get("codec_device", codec_device)
        self.worker_torch = ready.get("torch", "?")
        self.seed = seed

    # ---------------------------------------------------------------- low-level protocol
    def _drain_stderr(self) -> None:
        try:
            for line in iter(self.proc.stderr.readline, b""):
                self._stderr_tail.append(line)
        except Exception:  # noqa: BLE001  (pipe closed on exit)
            pass

    def _stderr_text(self) -> str:
        time.sleep(0.2)  # let the drain thread catch the worker's last words
        return b"".join(self._stderr_tail).decode("utf-8", errors="replace")

    def _raise_startup_error(self, msg: dict | None) -> None:
        try:
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        stderr = self._stderr_text()
        detail = (msg or {}).get("message", "no response from worker")
        gated_hint = ""
        if "GatedRepoError" in stderr or "GatedRepoError" in detail or "401" in stderr:
            gated_hint = ("\n\nThis Neuphonic model repo is gated: accept its terms at "
                          "https://huggingface.co/" + self.backbone_repo + " (and " + self.codec_repo +
                          ") while logged into your HF account, then run `hf auth login` "
                          "(or set HF_TOKEN) and try again.")
        raise NeuTTSWorkerError(f"NeuTTS worker failed to start: {detail}{gated_hint}\n\nstderr tail:\n{stderr[-1500:]}")

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _read_control_line(self, timeout: float | None = None) -> dict | None:
        """Reads one JSON line from stdout. Only used for the initial 'ready' handshake (which has no
        preceding chunk framing to worry about); the streaming path uses _read_event below."""
        line = self.proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _read_event(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            stderr = self._stderr_text()
            raise NeuTTSWorkerError(f"NeuTTS worker exited unexpectedly.\n\nstderr tail:\n{stderr[-1500:]}")
        return json.loads(line.decode("utf-8"))

    def _read_chunk_bytes(self, n_samples: int) -> np.ndarray:
        need = n_samples * 4
        buf = bytearray()
        while len(buf) < need:
            piece = self.proc.stdout.read(need - len(buf))
            if not piece:
                raise NeuTTSWorkerError("NeuTTS worker stdout closed mid-chunk")
            buf += piece
        return np.frombuffer(bytes(buf), dtype=np.float32)

    # ---------------------------------------------------------------- public interface
    def set_voice(self, spec: str) -> str:
        with self._lock:
            self._next_id += 1
            cid = self._next_id
            self._send({"cmd": "set_voice", "spec": spec, "id": cid})
            while True:
                msg = self._read_event()
                if msg.get("id") != cid:
                    continue
                if msg.get("event") == "error":
                    raise NeuTTSWorkerError(msg["message"])
                self.voice = msg["voice"]
                return self.voice

    def synth_stream(self, text: str, speed: float = 1.0):
        """Yields 24 kHz float32 chunks. If the caller abandons this generator early (barge-in), a
        `cancel` is sent on GeneratorExit and any in-flight response is drained so the pipe protocol
        stays synchronized for the next command."""
        with self._lock:
            self._next_id += 1
            cid = self._next_id
            self._send({"cmd": "synth_stream", "text": text, "id": cid})
            finished = False
            try:
                while True:
                    msg = self._read_event()
                    ev = msg.get("event")
                    if ev == "chunk":
                        yield self._read_chunk_bytes(int(msg["n"]))
                    elif ev in ("done", "cancelled") and msg.get("id") == cid:
                        finished = True
                        return
                    elif ev == "error":
                        finished = True
                        raise NeuTTSWorkerError(msg["message"])
            finally:
                if not finished:
                    # abandoned mid-stream (GeneratorExit from a barge-in): tell the worker to stop,
                    # then drain until its matching done/cancelled so the next command reads cleanly
                    try:
                        self._send({"cmd": "cancel"})
                        while True:
                            msg = self._read_event()
                            if msg.get("event") == "chunk":
                                self._read_chunk_bytes(int(msg["n"]))  # discard
                            elif msg.get("id") == cid:
                                break
                    except Exception:  # noqa: BLE001  worker may already be gone; nothing more to do
                        pass

    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        parts = list(self.synth_stream(text, speed))
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    def vram_mib(self) -> float | None:
        """The worker's torch has no CUDA (only llama.cpp does, independently); report the whole
        card's usage via nvidia-smi instead of a per-process torch figure."""
        try:
            import subprocess as sp
            out = sp.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            return float(out[0]) if out else None
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        """Orderly shutdown: stop any generation in flight, ask the worker to quit, give it 5 s to free
        the backbone, then kill it. Idempotent; also registered with atexit so an agent that exits
        without calling this still leaves no worker behind."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._send({"cmd": "cancel"})
            self._send({"cmd": "quit"})
            self.proc.stdin.close()
        except Exception:  # noqa: BLE001  (worker already gone)
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
                self.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
        try:
            atexit.unregister(self.close)
        except Exception:  # noqa: BLE001
            pass

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
