"""
transforms.py
-------------
Conversions between the fault toolkit's sample model and the tensors the
models consume.

**These now live in** :mod:`cpbench.data.samples`. They moved when
``w2cbench`` became the third package to need them and ``corabench`` was
found to hold a fourth, divergent implementation of the box conversion: a
degrees-versus-radians convention is exactly the kind of thing that must have
one definition, because a 57x error in a yaw target looks like a model that
simply will not converge rather than like a bug.

This module re-exports them so every existing import path in ``cobevtbench``
keeps working unchanged.
"""

from __future__ import annotations

from cpbench.data.samples import (EMPTY_BOXES, agent_to_ego_matrix,
                                  cooperative_gt_boxes, labels_to_array,
                                  ordered_agent_ids, world_to_ego_matrix)

__all__ = ["labels_to_array", "world_to_ego_matrix", "agent_to_ego_matrix",
           "ordered_agent_ids", "cooperative_gt_boxes", "EMPTY_BOXES"]
