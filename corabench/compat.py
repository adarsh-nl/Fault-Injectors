"""Torch version shims (A1 port, user-approved 2026-08-05).

corabench trains inside `opencood-official` (torch 1.12.1, py3.7) for full
pipeline identity with the three wrapped baselines, and validates in
`.venv-hpc` (torch 2.13). Every API that moved between those versions is
funnelled through here, so the port is one file instead of a scatter of
try/excepts.
"""

from __future__ import annotations

import torch
# torch.utils.checkpoint is a SUBMODULE: `import torch` alone does not
# bind it on torch 1.12, so checkpoint() below raised AttributeError
# whenever nothing else in the process had imported it first. The full
# training runs happened to work because OpenCOOD imports it; a
# minimal import check did not. Bind it explicitly.
import torch.utils.checkpoint  # noqa: F401

_TORCH2 = int(torch.__version__.split(".")[0]) >= 2


def autocast(device_type: str, enabled: bool = True):
    """torch.autocast in 2.x; torch.cuda.amp.autocast in 1.12."""
    if _TORCH2:
        return torch.autocast(device_type, enabled=enabled)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    import contextlib
    return contextlib.nullcontext()          # 1.12 CPU: no autocast


def grad_scaler(enabled: bool, init_scale: float = 0.0):
    """torch.amp.GradScaler in 2.x; torch.cuda.amp.GradScaler in 1.12.

    ``init_scale`` 0 means "torch default" (65536). It is a real constructor
    argument in both versions, so pass it properly rather than assigning to
    ``_init_scale`` after the fact -- the private attributes differ between
    1.12 and 2.x and the lazy-init path would silently ignore one of them.
    """
    kw = {"enabled": enabled}
    if init_scale:
        kw["init_scale"] = init_scale
    if _TORCH2:
        return torch.amp.GradScaler(**kw)
    return torch.cuda.amp.GradScaler(**kw)


def checkpoint(fn, *args):
    """use_reentrant=False where supported (2.x); 1.12 has only the
    reentrant form, which is fine for the deterministic chunk scan (no RNG,
    no kwargs, plain tensor args)."""
    if _TORCH2:
        return torch.utils.checkpoint.checkpoint(fn, *args,
                                                 use_reentrant=False)
    return torch.utils.checkpoint.checkpoint(fn, *args)


def load(path, map_location=None):
    """torch.load with the 2.x-only `weights_only` kwarg omitted on 1.12.

    `weights_only` arrived in torch 2.0. Passing it on 1.12 raises
    `TypeError: 'weights_only' is an invalid keyword argument for
    Unpickler()`, which is exactly how the resume verification (job 560245)
    died before it could compare anything. Same class of bug as the
    torch.utils.checkpoint submodule import above.
    """
    from cpbench.utils.torchio import load as _load   # single source
    return _load(path, map_location=map_location)


def no_autocast(device_type: str = "cuda"):
    """Context that genuinely DISABLES autocast inside it.

    Casting tensors to .float() is NOT sufficient: autocast intercepts the
    OPERATION, not the dtype, so ops on its cast-list (einsum, matmul, ...)
    downcast fp32 inputs back to fp16. The fp32 island therefore has to turn
    autocast off, not merely hand it fp32 tensors. (Caught by the pre-launch
    verification of the job-558108 fix: NaN was gone but inf remained.)
    """
    if _TORCH2:
        return torch.autocast(device_type, enabled=False)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    import contextlib
    return contextlib.nullcontext()
