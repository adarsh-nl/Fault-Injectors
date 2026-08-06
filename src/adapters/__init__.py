"""
adapters
--------
Per-framework/per-dataset glue between a third-party sample format and the
canonical ``CooperativeSample`` model the fault injectors run on.

An adapter is the ONLY place allowed to know a foreign schema. Injectors stay
dataset-agnostic, so a new dataset costs one adapter -- not a new injector and
not a rewrite.

    opencood  OpenCOODAdapter   OPV2V / V2XSet as loaded by OpenCOOD & V2X-ViT
    runtime   make_faulty_dataset(base_cls, spec)
"""

from .opencood import ModalityError, OpenCOODAdapter
from .runtime import FaultSpec, make_faulty_dataset

__all__ = ['ModalityError', 'OpenCOODAdapter', 'FaultSpec',
           'make_faulty_dataset']
