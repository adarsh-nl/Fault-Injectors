"""corabench -- CoRA (arXiv 2512.13191) reimplemented against docs/cora_spec.md.

Fresh 2026-08-05 rebuild (one-pass). The paper has no released code or
weights; every reconstructed choice is pre-decided in the spec and marked
"PAPER UNSPECIFIED" at its implementation site. Construction-time self-checks
live in `selfcheck.py`; `validate.py` is the synthetic forward/backward gate.
"""
from . import selfcheck  # noqa: F401
