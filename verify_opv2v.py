#!/usr/bin/env python3
"""
verify_opv2v.py
---------------
Phase 4 dataset verification for a CoRA/OpenCOOD training run.

Checks the on-disk OPV2V (or V2XSet) tree the way `src.datasets.OPV2VDataset`
and `corabench.scripts.common.build_adapters` will actually walk it, and
cross-checks the ego-selection convention against OpenCOOD's.

Run on a head node before submitting anything:

    python verify_opv2v.py --root /datasets/eemcs/ps/cv/opencood/opv2v
    python verify_opv2v.py --root ... --splits train validate test --deep

Exit code 0 = all checks passed, 1 = at least one BLOCKER.
No torch import, so it runs in a bare python3 with pyyaml.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")

BLOCKERS: list[str] = []
WARNINGS: list[str] = []


def blocker(msg: str) -> None:
    BLOCKERS.append(msg)
    print(f"  BLOCKER  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN     {msg}")


def ok(msg: str) -> None:
    print(f"  ok       {msg}")


def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def opencood_ego(cav_ids: list[str]) -> str:
    """OpenCOOD basedataset.py: lexicographic sort, infra (<0) pushed last."""
    lst = sorted(cav_ids)
    if lst and int(lst[0]) < 0:
        lst = lst[1:] + [lst[0]]
    return lst[0]


def adapter_ego(cav_ids: list[str]) -> str:
    """src/datasets/opv2v.py: numeric sort, first element."""
    return sorted(cav_ids, key=int)[0]


def check_split(root: Path, split: str, deep: bool) -> dict:
    print(f"\n--- split: {split} ---")
    split_dir = root / split
    if not split_dir.is_dir():
        blocker(f"split dir missing: {split_dir}")
        return {}

    scen_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not scen_dirs:
        blocker(f"no scenario dirs under {split_dir} "
                f"(build_adapters raises FileNotFoundError here)")
        return {}
    ok(f"{len(scen_dirs)} scenario dirs")

    total_frames = 0
    agent_hist: Counter = Counter()
    ego_mismatch: list[str] = []
    infra_as_ego: list[str] = []
    missing_pcd = 0
    empty_scen = 0
    ragged: list[str] = []

    for scen in scen_dirs:
        cav_dirs = [d for d in scen.iterdir()
                    if d.is_dir() and is_int(d.name)]
        cav_ids = [d.name for d in cav_dirs]
        if not cav_ids:
            blocker(f"{scen.name}: no numeric CAV folders "
                    f"(OPV2VDataset raises FileNotFoundError)")
            empty_scen += 1
            continue

        agent_hist[len(cav_ids)] += 1

        oc, ad = opencood_ego(cav_ids), adapter_ego(cav_ids)
        if oc != ad:
            ego_mismatch.append(f"{scen.name}: opencood={oc} adapter={ad} "
                                f"ids={sorted(cav_ids, key=int)}")
        if int(ad) < 0:
            infra_as_ego.append(scen.name)

        # frame counts, per the ADAPTER's ego (that is what will be indexed)
        ego_dir = scen / ad
        ts = sorted(p.stem for p in ego_dir.glob("*.yaml")
                    if not p.name.startswith("data_protocol"))
        if not ts:
            blocker(f"{scen.name}: adapter ego {ad} has no .yaml frames")
            continue
        total_frames += len(ts)

        counts = {}
        for cid in cav_ids:
            n = len([p for p in (scen / cid).glob("*.yaml")
                     if not p.name.startswith("data_protocol")])
            counts[cid] = n
        if len(set(counts.values())) > 1:
            ragged.append(f"{scen.name}: {counts}")

        if deep:
            for cid in cav_ids:
                for t in ts:
                    if (scen / cid / f"{t}.yaml").exists() and \
                       not (scen / cid / f"{t}.pcd").exists():
                        missing_pcd += 1

        # parse one yaml for schema
        if scen is scen_dirs[0]:
            with (ego_dir / f"{ts[0]}.yaml").open() as fh:
                params = yaml.safe_load(fh)
            for key in ("lidar_pose", "vehicles"):
                if key not in params:
                    blocker(f"{scen.name}/{ad}/{ts[0]}.yaml missing "
                            f"required key {key!r}")
            lp = params.get("lidar_pose")
            if lp is not None and len(lp) != 6:
                blocker(f"lidar_pose has {len(lp)} entries, expected 6 "
                        f"[x,y,z,roll,yaw,pitch]")
            else:
                ok("yaml schema: lidar_pose(6) + vehicles present")
            nveh = len(params.get("vehicles") or {})
            ok(f"sample frame lists {nveh} vehicles in the EGO yaml")

    ok(f"{total_frames} total ego frames "
       f"(= len(ConcatDataset), = samples per epoch)")
    ok(f"agents per scenario: "
       f"{dict(sorted(agent_hist.items()))}")

    if ego_mismatch:
        blocker(f"{len(ego_mismatch)} scenario(s) where the adapter picks a "
                f"DIFFERENT ego than OpenCOOD. Results are not comparable to "
                f"published numbers. First few:")
        for line in ego_mismatch[:5]:
            print(f"             {line}")
    else:
        ok("ego selection agrees with OpenCOOD in every scenario")

    if infra_as_ego:
        blocker(f"{len(infra_as_ego)} scenario(s) where the adapter makes the "
                f"INFRASTRUCTURE unit (negative id) the ego. OpenCOOD "
                f"explicitly forbids this. Scenarios: {infra_as_ego[:5]}")

    if ragged:
        warn(f"{len(ragged)} scenario(s) where agents have unequal frame "
             f"counts. The adapter drops non-ego agents on missing "
             f"timestamps, so agent count varies within the scenario. "
             f"First: {ragged[0]}")

    if deep and missing_pcd:
        blocker(f"{missing_pcd} frame(s) have a .yaml but no .pcd. "
                f"OPV2VDataset leaves frame.lidar = None and the agent is "
                f"then silently skipped by CoRADataset.")
    elif deep:
        ok("every yaml frame has a matching pcd")

    return {"scenarios": len(scen_dirs), "frames": total_frames,
            "agents": dict(agent_hist)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="dataset root containing train/ validate/ test/")
    ap.add_argument("--splits", nargs="*",
                    default=["train", "validate", "test"])
    ap.add_argument("--deep", action="store_true",
                    help="check every yaml has a matching pcd (slow, does "
                         "full directory walk)")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"root: {root}")
    if not root.is_dir():
        blocker(f"root does not exist: {root}")
        print("\nVERDICT: NO-GO")
        return 1
    ok("root exists")

    summary = {s: check_split(root, s, args.deep) for s in args.splits}

    print("\n=== summary ===")
    for split, info in summary.items():
        if info:
            print(f"  {split:9s} {info['scenarios']:3d} scenarios  "
                  f"{info['frames']:6d} frames  agents={info['agents']}")

    print(f"\nblockers: {len(BLOCKERS)}   warnings: {len(WARNINGS)}")
    print(f"VERDICT: {'NO-GO' if BLOCKERS else 'GO'}")
    return 1 if BLOCKERS else 0


if __name__ == "__main__":
    raise SystemExit(main())
