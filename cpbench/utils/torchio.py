"""Version-skew-safe ``torch.load``.

Lives in ``cpbench`` because more than one paper package needs it and the
paper packages are siblings that must never import each other
(``cpbench/tests/test_layering.py``). ``corabench.compat.load`` delegates here.

The skew this exists for bit twice, in opposite directions, within one day:

* **torch 1.12** (the ``opencood-official`` training env, py3.7) has no
  ``weights_only`` kwarg at all. Passing it raises
  ``TypeError: 'weights_only' is an invalid keyword argument for
  Unpickler()`` -- this killed the resume path of job 560245 before it could
  compare anything.
* **torch 2.6+** (``.venv-hpc`` runs 2.13) flipped ``weights_only`` to default
  ``True``. A checkpoint carrying numpy RNG state then fails with
  ``UnpicklingError: ... GLOBAL numpy.core.multiarray._reconstruct was not an
  allowed global`` -- this killed the *comparator* of job 561012 at the exact
  step that checks the GradScaler state.

Fixing only one side is what turned two consecutive verification runs into
two different crashes, so every ``torch.load`` in the repo routes here.

``weights_only=False`` is correct for this project's checkpoints: they are
produced by our own training jobs on our own cluster and deliberately contain
non-tensor state (RNG streams, scheduler and scaler dicts) that the restricted
unpickler rejects by design.
"""

from __future__ import annotations

import torch

_TORCH2 = int(torch.__version__.split(".")[0]) >= 2


def load(path, map_location=None):
    """``torch.load`` that works on both torch 1.12 and torch 2.x."""
    if _TORCH2:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)
