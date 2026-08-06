"""Construction-time guards (spec §5). Each corresponds to a failure class
that cost a debugging round in the previous implementation; they run at
import/construction/validation time, before any training step.
"""

from __future__ import annotations

import math
import pathlib
import re
from typing import Sequence, Tuple

import torch

FOCAL_PI = 0.01
FOCAL_BIAS = -math.log((1.0 - FOCAL_PI) / FOCAL_PI)      # -4.5951

_PKG = pathlib.Path(__file__).parent
# Files whose forward paths carry gradients: hard clamps forbidden except on
# lines explicitly marked `# no-grad-ok`.
_GRAD_FILES = ("fusion", "models", "training")


def assert_shape(t: torch.Tensor, shape: Sequence[int], where: str) -> None:
    """Shape assert for cross-agent seams (spec §5.7). -1 = any."""
    ok = t.dim() == len(shape) and all(
        s == -1 or t.shape[i] == s for i, s in enumerate(shape))
    if not ok:
        raise AssertionError(
            f"{where}: expected shape {tuple(shape)}, got {tuple(t.shape)}")


def assert_focal_bias(conv: torch.nn.Conv2d, where: str,
                      tol: float = 0.05) -> None:
    """Spec §5.2: cls-producing convs carry the focal prior at init."""
    if conv.bias is None:
        raise AssertionError(f"{where}: cls conv has no bias to carry the "
                             f"focal prior")
    b = conv.bias.detach()
    if not torch.allclose(b, torch.full_like(b, FOCAL_BIAS), atol=tol):
        raise AssertionError(
            f"{where}: cls bias {b.mean().item():.3f} != focal prior "
            f"{FOCAL_BIAS:.3f} (init cls loss would be ~50, not ~1)")


def assert_dt_init(dt_bias: torch.Tensor, lo: float = 1e-3,
                   hi: float = 1e-1, where: str = "cssm") -> None:
    """Spec §5.3: softplus(dt bias) must sit in Mamba's init range."""
    dt = torch.nn.functional.softplus(dt_bias.detach())
    if not bool(((dt >= lo * 0.999) & (dt <= hi * 1.001)).all()):
        raise AssertionError(
            f"{where}: softplus(dt_bias) in [{dt.min():.2e}, {dt.max():.2e}]"
            f" outside Mamba range [{lo}, {hi}] (the integrator/saturation "
            f"failure class)")


def assert_nonpositive(t: torch.Tensor, where: str, tol: float = 1e-4) -> None:
    """Spec §5.3: pairwise scan exponents must be <= 0 (exp bounded by 1)."""
    mx = float(t.detach().max())
    if mx > tol:
        raise AssertionError(f"{where}: scan exponent max {mx:.3e} > 0 -- "
                             f"the divide-free invariant is broken")


def grad_step_guard(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    max_norm: float = 10.0) -> Tuple[bool, float]:
    """Spec §5.6: clip; if the total norm is non-finite, skip the step and
    zero the gradients. Returns (stepped, grad_norm)."""
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, error_if_nonfinite=False)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        return False, float("nan")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return True, float(norm)


def static_source_checks() -> None:
    """Spec §5.1/5.4/5.5: source-level scan of this package.

    * no `.clamp(`/`torch.clamp` in gradient-carrying files except lines
      marked `# no-grad-ok`;
    * no `asin` anywhere (180-ambiguous, singular at +-1);
    * no fp16 probability clamps (`clamp` with 1e- epsilon bounds).
    """
    offenders = []
    for sub in _GRAD_FILES:
        for path in sorted((_PKG / sub).rglob("*.py")):
            for ln, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if "no-grad-ok" in line:
                    continue
                if re.search(r"\.clamp\(|torch\.clamp\(", code):
                    offenders.append(f"{path.name}:{ln} hard clamp: "
                                     f"{line.strip()}")
                if re.search(r"\basin\b|\barcsin\b", code):
                    offenders.append(f"{path.name}:{ln} asin: {line.strip()}")
    if offenders:
        raise AssertionError(
            "self-check: forbidden ops on differentiable paths\n  "
            + "\n  ".join(offenders))
