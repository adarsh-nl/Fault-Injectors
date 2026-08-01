"""Keymatch diagnostic: official checkpoint state_dict vs this repo's reimpl.

Read-only, CPU-only. For each of the four benches it

  1. torch.loads the official released checkpoint and dumps key -> shape,
  2. instantiates the reimplementation from its shipped config and dumps
     key -> shape (recording the traceback instead if construction fails),
  3. compares the two key sets.

Outputs, all under ``results/diag/``:
    <model>_official_keys.txt   <model>_reimpl_keys.txt   keymatch_raw.json

The markdown summaries are rendered from ``keymatch_raw.json`` separately so
that this file stays a pure measurement and the prose stays reviewable.

Nothing here trains, evaluates, injects a fault, or writes outside
``results/diag/``.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
DIAG = REPO / "results" / "diag"
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", DIAG / "_ckpt"))
DIAG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

# Official released weights. ``None`` = no released weights exist at all,
# which is itself the finding (CoRA has neither released code nor weights).
OFFICIAL = {
    "cobevt": CKPT_ROOT / "cobevt" / "net_epoch60.pth",
    "v2xvit": CKPT_ROOT / "v2xvit" / "net_epoch60.pth",
    "where2comm": (CKPT_ROOT / "where2comm" / "point_pillar_where2comm_v2xset"
                   / "net_epoch50.pth"),
    "cora": None,
}

# Secondary checkpoints found in the OPV2V zoo; dumped for the inventory but
# compared only where a reimpl track exists for them.
EXTRA = {
    "cobevt_camera_dynamic": (
        CKPT_ROOT / "opv2v_zoo" / "cobevt_camera_dynamic" / "net_epoch91.pth",
        ("cobevtbench", ["model=cobevt_camera_dynamic",
                         "dataset=opv2v_camera"])),
    "cobevt_camera_static": (
        CKPT_ROOT / "opv2v_zoo" / "cobevt_camera_static" / "net_epoch91.pth",
        ("cobevtbench", ["model=cobevt_camera_static",
                         "dataset=opv2v_camera"])),
    "cobevt_lidar_opv2v_nocomp": (
        CKPT_ROOT / "opv2v_zoo" / "cobevt_lidar_opv2v_nocomp" / "net_epoch19.pth",
        ("cobevtbench", ["model=cobevt_lidar", "dataset=opv2v_lidar"])),
}

# The reimplementation to build for each model, and the config overrides that
# select the instantiation the checkpoint actually is.
REIMPL = {
    "cobevt": ("cobevtbench", ["model=cobevt_lidar", "dataset=opv2v_lidar"]),
    "v2xvit": ("v2xvitbench", ["model=v2xvit", "dataset=v2xset"]),
    "where2comm": ("w2cbench", ["model=where2comm_lidar", "dataset=v2xset"]),
    "cora": ("corabench", ["model=cora", "dataset=opv2v"]),
}


def load_official(path: Path) -> dict:
    """key -> shape for a released checkpoint, unwrapping a training wrapper."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and not any(torch.is_tensor(v)
                                         for v in obj.values()):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj:
                obj = obj[k]
                break
    return {k: tuple(v.shape) for k, v in obj.items() if torch.is_tensor(v)}


def build_reimpl(pkg: str, overrides: list) -> dict:
    mod = __import__(f"{pkg}.scripts.common", fromlist=["common"])
    model = mod.build_model(mod.load(overrides))
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def dump(path: Path, d: dict, header: str) -> None:
    n_params = sum(int(torch.tensor(list(s)).prod()) if s else 1
                   for s in d.values())
    lines = [header, f"# {len(d)} tensors, {n_params} parameters", ""]
    lines += [f"{k}\t{list(s)}" for k, s in d.items()]
    path.write_text("\n".join(lines) + "\n")


def prefixes(keys, depth: int = 2):
    c = Counter()
    for k in keys:
        c[".".join(k.split(".")[:depth])] += 1
    return c.most_common(30)


def compare(off: dict, re_: dict) -> dict:
    off_keys, re_keys = set(off), set(re_)
    exact = {k for k in off_keys & re_keys if off[k] == re_[k]}
    name_shape_diff = {k for k in off_keys & re_keys if off[k] != re_[k]}
    off_only, re_only = off_keys - re_keys, re_keys - off_keys

    # An upper bound on what a pure rename converter could ever recover: how
    # many official tensors have SOME unclaimed reimpl tensor of identical
    # shape. It ignores whether the pairing is semantically correct, so it
    # overstates -- which is the safe direction for a "not worth it" verdict.
    off_shapes = Counter(off[k] for k in off_only)
    re_shapes = Counter(re_[k] for k in re_only)
    pairable = sum(min(off_shapes[s], re_shapes[s]) for s in off_shapes)

    return {
        "official_tensors": len(off),
        "reimpl_tensors": len(re_),
        "exact_name_and_shape": len(exact),
        "pct_official_exact": round(100.0 * len(exact) / max(len(off), 1), 2),
        "name_match_shape_differs": len(name_shape_diff),
        "official_only": len(off_only),
        "reimpl_only": len(re_only),
        "shape_pairable_among_unmatched": pairable,
        "pct_official_shape_recoverable_upper_bound": round(
            100.0 * (len(exact) + pairable) / max(len(off), 1), 2),
        "examples_exact": sorted(exact)[:10],
        "examples_name_match_shape_differs": [
            {"key": k, "official": list(off[k]), "reimpl": list(re_[k])}
            for k in sorted(name_shape_diff)[:10]],
        "examples_official_only": sorted(off_only)[:20],
        "examples_reimpl_only": sorted(re_only)[:20],
        "official_prefixes": prefixes(off),
        "reimpl_prefixes": prefixes(re_),
    }


def one(name: str, ckpt, reimpl_spec) -> dict:
    entry: dict = {"model": name}
    print(f"### {name}", flush=True)

    off = None
    if ckpt is None:
        entry["official"] = {"status": "ABSENT",
                             "reason": "no released weights exist"}
    elif not Path(ckpt).exists():
        entry["official"] = {"status": "ABSENT", "reason": f"missing {ckpt}"}
    else:
        try:
            off = load_official(Path(ckpt))
            dump(DIAG / f"{name}_official_keys.txt", off,
                 f"# official checkpoint: {ckpt}")
            entry["official"] = {"status": "OK", "tensors": len(off),
                                 "path": str(ckpt)}
            print(f"  official: {len(off)} tensors", flush=True)
        except Exception:
            entry["official"] = {"status": "ERROR",
                                 "traceback": traceback.format_exc()}
            print("  official LOAD FAILED", flush=True)

    re_ = None
    pkg, ov = reimpl_spec
    try:
        re_ = build_reimpl(pkg, ov)
        dump(DIAG / f"{name}_reimpl_keys.txt", re_,
             f"# reimpl: {pkg} {' '.join(ov)}")
        entry["reimpl"] = {"status": "OK", "tensors": len(re_), "pkg": pkg,
                           "overrides": ov}
        print(f"  reimpl:   {len(re_)} tensors", flush=True)
    except Exception:
        entry["reimpl"] = {"status": "BLOCKED-BY-IMPORT", "pkg": pkg,
                           "overrides": ov,
                           "traceback": traceback.format_exc()}
        print("  reimpl BUILD FAILED", flush=True)

    if off and re_:
        entry["keymatch"] = compare(off, re_)
        km = entry["keymatch"]
        print(f"  exact name+shape: {km['exact_name_and_shape']}/"
              f"{km['official_tensors']} ({km['pct_official_exact']}%)",
              flush=True)
    return entry


def main() -> None:
    report = {}
    for name in ("cobevt", "v2xvit", "where2comm", "cora"):
        report[name] = one(name, OFFICIAL[name], REIMPL[name])
    for name, (ckpt, spec) in EXTRA.items():
        report[name] = one(name, ckpt, spec)
    (DIAG / "keymatch_raw.json").write_text(json.dumps(report, indent=2))
    print("WROTE", DIAG / "keymatch_raw.json", flush=True)


if __name__ == "__main__":
    main()
