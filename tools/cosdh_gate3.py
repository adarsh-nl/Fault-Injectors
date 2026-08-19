"""Gate 3: pose fire-check WITH magnitude and frame verification.

A naive fire-check would have passed BOTH bugs found while building this
adapter, so this checks four things, not one:

  A. MAGNITUDE  -- measured displacement in the derived matrices vs theory,
                   in BOTH `transformation_matrix` (late branch) and
                   `pairwise_t_matrix` (feature warping). They derive from the
                   pose INDEPENDENTLY, so a wrong-frame injection shows up in
                   one and not the other.
  B. PER-FRAME SPREAD -- offsets must DIFFER across frames. Seeding on a
                   constant gives every frame the identical offset: a constant
                   bias that passes any single-frame magnitude check.
  C. CLEAN SLOT -- `lidar_pose_clean` must be byte-identical to its
                   pre-injection value. If the perturbation leaks into it,
                   both derived matrices agree and the fault is inert.
  D. YAW INDEX  -- verified through the RESULTING ROTATION, not the array
                   slot: index 4 is yaw in this convention, and checking the
                   slot would not catch a wrong-index write that happens to
                   land somewhere plausible.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

import opencood                                                # noqa: E402
assert "cosdh-official" in opencood.__file__, opencood.__file__

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencood.hypes_yaml import yaml_utils                     # noqa: E402
from opencood.data_utils.datasets import OPV2VBaseDataset      # noqa: E402
from opencood.data_utils.datasets import getIntermediatelateFusionDataset  # noqa: E402
from opencood.utils import pose_utils                          # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2, x_to_world  # noqa: E402

from src.adapters.cosdh import make_pose_noise_hook, _FRAME     # noqa: E402
from src.adapters.runtime import FaultSpec                      # noqa: E402


def yaw_of(T):
    """Extract yaw from a 4x4 rigid transform."""
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--sigma_xy", type=float, default=0.2)
    ap.add_argument("--sigma_yaw", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, "config.yaml"))
    ds = getIntermediatelateFusionDataset(OPV2VBaseDataset)(
        params=hypes, visualize=False, train=False)

    spec = FaultSpec(seed=1234,
                     pose_error={"sigma_xy": args.sigma_xy,
                                 "sigma_heading": args.sigma_yaw})
    hook = make_pose_noise_hook(pose_utils.add_noise_data_dict, spec)

    rows, clean_ok, n_pairs = [], True, 0
    for idx in range(min(args.frames, len(ds))):
        base = ds.retrieve_base_data(idx)
        _FRAME["idx"] = idx
        pre = copy.deepcopy(base)
        pre = pose_utils.add_noise_data_dict(pre, hypes["noise_setting"])
        post = hook(copy.deepcopy(base), hypes["noise_setting"])

        ego = [c for c in post if post[c].get("ego")][0]
        for cav in post:
            if cav == ego:
                continue
            n_pairs += 1
            # C. clean slot untouched
            if not np.allclose(np.asarray(pre[cav]["params"]["lidar_pose_clean"],
                                          dtype=float),
                               np.asarray(post[cav]["params"]["lidar_pose_clean"],
                                          dtype=float), atol=0, rtol=0):
                clean_ok = False
            p0 = np.asarray(pre[cav]["params"]["lidar_pose"], float)
            p1 = np.asarray(post[cav]["params"]["lidar_pose"], float)
            # A. derived matrices, both paths
            T0 = x1_to_x2(list(p0), list(np.asarray(
                pre[ego]["params"]["lidar_pose"], float)))
            T1 = x1_to_x2(list(p1), list(np.asarray(
                post[ego]["params"]["lidar_pose"], float)))
            d_tm = float(np.linalg.norm(T1[:2, 3] - T0[:2, 3]))
            W0, W1 = x_to_world(list(p0)), x_to_world(list(p1))
            d_pw = float(np.linalg.norm(W1[:2, 3] - W0[:2, 3]))
            rows.append({
                "idx": idx, "cav": str(cav),
                "d_pose_xy": float(np.hypot(*(p1 - p0)[:2])),
                "d_transformation_matrix": d_tm,
                "d_pairwise_world": d_pw,
                "d_yaw_pose_deg": float((p1 - p0)[4]),
                "d_yaw_matrix_deg": yaw_of(T1) - yaw_of(T0),
                "delta_slots": [float(x) for x in (p1 - p0)],
            })

    arr = lambda k: np.array([r[k] for r in rows])             # noqa: E731
    theory = args.sigma_xy * np.sqrt(np.pi / 2) * np.sqrt(2) / np.sqrt(2)
    res = {
        "n_frames": len(set(r["idx"] for r in rows)),
        "n_pairs": n_pairs,
        "A_magnitude": {
            "mean_d_pose_xy": float(arr("d_pose_xy").mean()),
            "mean_d_transformation_matrix": float(arr("d_transformation_matrix").mean()),
            "mean_d_pairwise_world": float(arr("d_pairwise_world").mean()),
            "theory_rayleigh_mean": float(args.sigma_xy * np.sqrt(np.pi / 2)),
            "sigma_xy": args.sigma_xy,
        },
        "B_per_frame_spread": {
            "unique_d_pose_xy": int(len(np.unique(np.round(arr("d_pose_xy"), 9)))),
            "std_d_pose_xy": float(arr("d_pose_xy").std()),
            "min": float(arr("d_pose_xy").min()),
            "max": float(arr("d_pose_xy").max()),
            "IDENTICAL_ACROSS_FRAMES": bool(arr("d_pose_xy").std() < 1e-12),
        },
        "C_clean_slot_untouched": bool(clean_ok),
        "D_yaw": {
            "mean_d_yaw_pose_deg": float(arr("d_yaw_pose_deg").mean()),
            "std_d_yaw_pose_deg": float(arr("d_yaw_pose_deg").std()),
            "mean_abs_d_yaw_matrix_deg": float(np.abs(arr("d_yaw_matrix_deg")).mean()),
            "sigma_yaw_deg": args.sigma_yaw,
            "slots_touched": sorted(set(
                i for r in rows for i, v in enumerate(r["delta_slots"])
                if abs(v) > 1e-12)),
        },
    }
    with open(os.path.join(args.out, "gate3.json"), "w") as fh:
        json.dump({"summary": res, "rows": rows[:200]}, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
