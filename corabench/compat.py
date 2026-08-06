"""Torch version shims (A1 port, user-approved 2026-08-05).

corabench trains inside `opencood-official` (torch 1.12.1, py3.7) for full
pipeline identity with the three wrapped baselines, and validates in
`.venv-hpc` (torch 2.13). Every API that moved between those versions is
funnelled through here, so the port is one file instead of a scatter of
try/excepts.
"""

from __future__ import annotations

import torch

_TORCH2 = int(torch.__version__.split(".")[0]) >= 2


def autocast(device_type: str, enabled: bool = True):
    """torch.autocast in 2.x; torch.cuda.amp.autocast in 1.12."""
    if _TORCH2:
        return torch.autocast(device_type, enabled=enabled)
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    import contextlib
    return contextlib.nullcontext()          # 1.12 CPU: no autocast


def grad_scaler(enabled: bool):
    """torch.amp.GradScaler in 2.x; torch.cuda.amp.GradScaler in 1.12."""
    if _TORCH2:
        return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def checkpoint(fn, *args):
    """use_reentrant=False where supported (2.x); 1.12 has only the
    reentrant form, which is fine for the deterministic chunk scan (no RNG,
    no kwargs, plain tensor args)."""
    if _TORCH2:
        return torch.utils.checkpoint.checkpoint(fn, *args,
                                                 use_reentrant=False)
    return torch.utils.checkpoint.checkpoint(fn, *args)
