"""Shared helpers for the Parakeet / Silero benchmark scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "audio"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = os.environ.get("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3")


def preload_cuda_dlls() -> None:
    """onnxruntime-gpu on Windows needs the CUDA/cuDNN DLLs on the search path.
    onnxruntime>=1.21 can pull them from the nvidia-* pip packages."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


def gpu_mem_mib() -> tuple[int, int] | None:
    """(used, total) MiB for GPU 0 via NVML, or None if NVML unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return m.used // 2**20, m.total // 2**20
    except Exception:
        return None


def proc_gpu_mem_mib() -> int | None:
    """GPU memory used by THIS process (MiB), via NVML per-process query."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == os.getpid() and p.usedGpuMemory:
                return p.usedGpuMemory // 2**20
    except Exception:
        pass
    return None


def load_parakeet(device: str = "cuda", quantization: str | None = None):
    """Load Parakeet via onnx-asr and assert the requested provider is really active.

    device: "cuda"   -> encoder + decoder_joint both on CUDA (onnx-asr default)
            "hybrid" -> encoder on CUDA, decoder_joint on CPU (the per-frame decoder loop is
                        launch-bound on GPU; CPU avoids ~T host<->device round trips)
            "cpu"    -> everything on CPU
    """
    import onnx_asr
    import onnxruntime as ort

    hybrid = device == "hybrid"
    if hybrid:
        device = "cuda"
    if device == "cuda":
        preload_cuda_dlls()
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            sys.exit(f"FATAL: CUDAExecutionProvider not available. Providers: {avail}")
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    t0 = time.perf_counter()
    model = onnx_asr.load_model(MODEL_NAME, quantization=quantization, providers=providers)
    if hybrid:
        _move_decoder_to_cpu(model, int8=os.environ.get("HYBRID_DEC_INT8", "1") == "1",
                             threads=int(os.environ.get("HYBRID_DEC_THREADS", "0")))
    # ORT's Python wrapper silently rebuilds the session on CPU if a CUDA kernel fails at run time.
    # We want that to be a hard error, not a hidden 100x slowdown.
    for sess in _sessions(model):
        sess.disable_fallback()
    load_s = time.perf_counter() - t0

    # Verify the encoder session actually got CUDA (ORT silently falls back to CPU otherwise).
    active = _active_providers(model)
    if device == "cuda" and not any("CUDA" in p for p in active):
        sys.exit(f"FATAL: model loaded but CUDA is NOT active. Active providers: {active}")
    return model, load_s, active


def _move_decoder_to_cpu(model, int8: bool = True, threads: int = 0) -> None:
    """Rebuild the transducer decoder_joint session on CPU.

    The decoder runs once per encoder frame (12.5 frames/s of audio) with tiny tensors, so it is
    launch-bound on GPU. On CPU the int8 decoder is the fastest we measured (~0.2 ms/frame)."""
    import onnxruntime as ort

    asr = model.asr
    sess = getattr(asr, "_decoder_joint", None)
    if sess is None:
        return
    path = Path(sess._model_path)  # set by ORT's InferenceSession
    if int8:
        cand = path.with_name(path.name.replace(".onnx", ".int8.onnx"))
        if not cand.exists():
            try:
                from huggingface_hub import hf_hub_download

                cand = Path(hf_hub_download("istupakov/parakeet-tdt-0.6b-v3-onnx", cand.name))
            except Exception:
                cand = path  # fall back to the fp32 decoder on CPU
        path = cand
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    asr._decoder_joint = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    asr._decoder_joint_path = str(path)


def _sessions(model) -> list:
    """Walk the adapter object graph and collect ORT InferenceSession objects."""
    import onnxruntime as ort

    found, stack, visited = [], [model], set()
    while stack:
        obj = stack.pop()
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        if isinstance(obj, ort.InferenceSession):
            found.append(obj)
            continue
        for attr in getattr(obj, "__dict__", {}).values():
            if hasattr(attr, "__dict__") or isinstance(attr, ort.InferenceSession):
                stack.append(attr)
    return found


def _active_providers(model) -> list[str]:
    seen: set[str] = set()
    for s in _sessions(model):
        seen.update(s.get_providers())
    return sorted(seen)


def assert_cuda_still_active(model, device: str) -> list[str]:
    """Call after the warm-up inference: providers can change if ORT fell back to CPU."""
    active = _active_providers(model)
    if device in ("cuda", "hybrid") and not any("CUDA" in p for p in active):
        sys.exit(f"FATAL: CUDA dropped after first inference (fallback to CPU). Active: {active}")
    return active


def load_wav_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        import math

        # simple linear resample; good enough for a benchmark harness
        n = int(math.ceil(len(data) * 16000 / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def normalize_text(s: str) -> str:
    import re

    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return " ".join(s.split())


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.wer(r, h))


def pct(a: list[float], p: float) -> float:
    return float(np.percentile(a, p)) if a else float("nan")


def save_json(name: str, obj) -> Path:
    out = RESULTS_DIR / name
    out.write_text(json.dumps(obj, indent=2, default=str))
    return out


def wasapi_settings(device):
    """WASAPI shared mode won't give 16 kHz on a 48 kHz mic unless auto_convert is on."""
    import sounddevice as sd
    try:
        dev = sd.query_devices(device if device is not None else sd.default.device[0])
        if "WASAPI" in sd.query_hostapis(dev["hostapi"])["name"]:
            return sd.WasapiSettings(auto_convert=True)
    except Exception:
        pass
    return None


def resolve_input_device(spec):
    """Turn --device into a stable sounddevice index.

    Windows renumbers audio devices between sessions, so a bare index (52) can silently point at a
    different device tomorrow. Accept: None (system default), an int (used as-is), or a name substring
    like "Yeti" (case-insensitive; prefers the WASAPI entry). Prints what it picked.
    """
    import sounddevice as sd

    if spec is None:
        idx = sd.default.device[0]
    elif isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
    else:
        apis = sd.query_hostapis()
        hits = [(i, d) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower()]
        if not hits:
            raise SystemExit(f"no input device matching '{spec}'. Run bench/list_devices.py")
        hits.sort(key=lambda t: 0 if "WASAPI" in apis[t[1]["hostapi"]]["name"] else 1)
        idx = hits[0][0]
    d = sd.query_devices(idx)
    print(f"input device {idx}: {d['name']} [{sd.query_hostapis(d['hostapi'])['name']}] {d['default_samplerate']:.0f} Hz")
    return idx
