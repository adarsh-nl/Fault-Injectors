#!/usr/bin/env python3
"""Dump key -> shape from a released `.pth` **without importing torch**.

Why this exists: on this cluster `import torch` from the NFS-mounted
`.venv-hpc` (5.4 GB) takes upwards of fifteen minutes under load, which makes
even a read-only diagnostic a batch job (docs/debug_log.md DL-001). But a
checkpoint saved by `torch.save` is just a ZIP whose `data.pkl` member is a
pickle, and the shape of every tensor is recorded *in that pickle* as the
arguments to `torch._utils._rebuild_tensor_v2`. Stubbing the handful of torch
symbols the pickle names is enough to read the whole layout with the standard
library alone, in well under a second.

This reads only the metadata pickle. Tensor storages are never touched, so
nothing here can produce a number -- it answers "what modules and shapes does
this checkpoint contain", which is the whole official side of a keymatch.

    python3 results/diag/dump_official_keys.py <ckpt.pth> [more.pth ...]
"""
from __future__ import annotations

import io
import pickle
import sys
import zipfile
from collections import Counter
from pathlib import Path

DIAG = Path(__file__).resolve().parent


class _Stub:
    """Stand-in for any torch object the pickle names."""

    def __init__(self, *a, **k):
        self.args = a

    def __call__(self, *a, **k):
        return _Stub(*a)

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (_Stub, ())


class _Tensor:
    """What `_rebuild_tensor_v2(storage, offset, size, stride, ...)` builds."""

    def __init__(self, *args):
        self.size = tuple(args[2]) if len(args) > 2 else ()
        self.storage = args[0] if args else None

    def __repr__(self):
        return f"Tensor{list(self.size)}"


def _rebuild_tensor_v2(*args):
    return _Tensor(*args)


def _rebuild_parameter(data, *_):
    return data


class _Unpickler(pickle.Unpickler):
    """Resolve torch names to stubs; refuse to import anything real."""

    def find_class(self, module, name):
        if module.startswith("torch"):
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _rebuild_tensor_v2
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            return _Stub
        if module in ("collections", "__builtin__", "builtins"):
            return super().find_class(module, name)
        if module.startswith("numpy"):
            return _Stub
        return _Stub

    def persistent_load(self, pid):
        # storages arrive as persistent ids; we never read their bytes
        return _Stub(pid)


def dtype_of(t):
    """Best-effort dtype from the storage persistent id, else unknown."""
    try:
        pid = t.storage.args[0]
        for part in pid:
            s = str(part)
            if "Storage" in s or s in ("float", "double", "half", "long",
                                       "int", "short", "char", "byte", "bool"):
                return s
    except Exception:
        pass
    return "?"


def load_layout(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist()
                      if n.endswith("data.pkl") or n.endswith(".pkl"))
        obj = _Unpickler(io.BytesIO(z.read(member))).load()

    # unwrap a training wrapper: {'model': state_dict, 'optimizer': ...}
    if isinstance(obj, dict) and not any(isinstance(v, _Tensor)
                                         for v in obj.values()):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    return {k: v.size for k, v in obj.items() if isinstance(v, _Tensor)}


def prefixes(keys, depth=2):
    c = Counter()
    for k in keys:
        c[".".join(str(k).split(".")[:depth])] += 1
    return c.most_common(40)


def main(paths):
    for p in paths:
        p = Path(p)
        layout = load_layout(p)
        n_params = sum(
            __import__("math").prod(s) if s else 1 for s in layout.values())
        name = p.parent.name if p.parent.name not in ("", ".") else p.stem
        out = DIAG / f"{name}_official_keys.txt"
        lines = [f"# official checkpoint: {p}",
                 f"# {len(layout)} tensors, {n_params} parameters",
                 "# dumped WITHOUT torch (stdlib zipfile + pickle)", ""]
        lines += [f"{k}\t{list(s)}" for k, s in layout.items()]
        lines += ["", "# top-level module prefixes (depth 2)"]
        lines += [f"# {pre}\t{n}" for pre, n in prefixes(layout)]
        out.write_text("\n".join(lines) + "\n")
        print(f"{name:32s} {len(layout):4d} tensors  {n_params:>12,d} params"
              f"  -> {out.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
