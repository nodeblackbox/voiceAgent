"""Compatibility shims for the NeuTTS dependency chain, applied before `import neutts`.

`neucodec` imports `torchtune.modules.RotaryPositionalEmbeddings` at module load time just to build its
Vocos decoder variant; `torchtune` (latest: 0.6.1) in turn references `torchao.dtypes.nf4tensor.NF4Tensor`
at ITS module load time — a class current `torchao` (0.18.0+) has since removed as part of an internal
refactor. We never load an NF4-quantized checkpoint (that's a QLoRA fine-tuning feature, unrelated to
running NeuTTS for inference), so a dummy placeholder that satisfies the import is safe: the class is
never actually instantiated on our code path. Same pattern the project's own `neutts.py` already uses
for the `resemble-perth` / `pkg_resources` gap — a real, if awkward, way real-world dependency trees drift.

Call `apply()` once, before the first `import neutts` (or anything importing `neucodec`/`torchtune`).
"""
from __future__ import annotations

import sys
import types


def apply() -> None:
    try:
        import torchao.dtypes.nf4tensor  # noqa: F401
        return  # real module has it; nothing to shim
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import torchao.dtypes as _dtypes
    except (ImportError, ModuleNotFoundError):
        # torchao isn't installed at all in this interpreter -- register bare placeholder
        # packages instead of crashing, so `import torchao.dtypes.nf4tensor` still succeeds.
        _torchao = sys.modules.get("torchao") or types.ModuleType("torchao")
        sys.modules.setdefault("torchao", _torchao)
        _dtypes = types.ModuleType("torchao.dtypes")
        sys.modules["torchao.dtypes"] = _dtypes
        _torchao.dtypes = _dtypes

    shim = types.ModuleType("torchao.dtypes.nf4tensor")

    def _unused(name):
        def _fn(*a, **k):
            raise NotImplementedError(
                f"torchao.dtypes.nf4tensor.{name} is a compatibility placeholder (nf4tensor was removed "
                "from installed torchao, an unrelated QLoRA fine-tuning feature); NeuTTS inference never "
                "calls it."
            )
        return _fn

    class NF4Tensor:  # only needs to exist for torchtune's import + isinstance checks to succeed
        def __init__(self, *a, **k):
            _unused("NF4Tensor")()

    shim.NF4Tensor = NF4Tensor
    shim.linear_nf4 = _unused("linear_nf4")
    shim.to_nf4 = _unused("to_nf4")
    shim.implements = lambda *a, **k: (lambda fn: fn)  # decorator factory: registers a torch_function impl
    sys.modules["torchao.dtypes.nf4tensor"] = shim
    _dtypes.nf4tensor = shim
