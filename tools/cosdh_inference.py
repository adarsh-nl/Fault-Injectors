"""CoSDH inference driver with optional fault injection.

Mirrors ``tools/fi_inference.py`` in shape, but CoSDH is an OpenCOOD *fork*
and diverges at every hook point, so it needs its own driver.

Two interception points, and only two:

1. ``opencood.tools.inference.build_dataset`` is replaced in that module's
   namespace so the OPV2V *base* class can be swapped for our faulty subclass
   before the class factory consumes it. ``build_dataset`` resolves both the
   fusion wrapper and the base class by ``eval()`` on constructed name
   strings, so there is no seam to pass a class in -- the six lines are
   reimplemented here instead, with the base class wrapped.

2. ``add_noise_data_dict`` is replaced in the FUSION DATASET's namespace for
   the pose fault. It must run AFTER CoSDH's own copy of
   ``lidar_pose -> lidar_pose_clean``; see ``src/adapters/cosdh.py``.

Everything downstream of dataset construction -- ``inference_late_fusion``,
``post_process``, ``eval_utils.caluclate_tp_fp``, ``eval_final_results``, NMS
-- runs from the UNMODIFIED ``inference.main()``, so AP and NMS stay
byte-identical to the official pipeline.

    python tools/cosdh_inference.py --model_dir DIR --fi_out OUT \
        [--fi_spec '{"pose_error": {"sigma_xy": 0.2, "sigma_heading": 0.2}}']
"""

import argparse
import json
import os
import sys

# The cosdh conda env carries an INHERITED EDITABLE INSTALL of vanilla
# OpenCOOD, so `import opencood` resolves to ~/opencood-official unless
# PYTHONPATH puts the fork first. Running CoSDH's checkpoint against vanilla
# code would be silent and catastrophic, so this is an assert, not a comment.
import opencood                                                # noqa: E402

if "cosdh-official" not in opencood.__file__:
    raise SystemExit(
        "FATAL: 'opencood' resolves to %s\n"
        "Expected the CoSDH fork. Set PYTHONPATH=$HOME/cosdh-official BEFORE "
        "the inherited editable vanilla install." % opencood.__file__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencood.data_utils.datasets import (                     # noqa: E402
    getLateFusionDataset, getEarlyFusionDataset,
    getIntermediateFusionDataset, getIntermediate2stageFusionDataset,
    getIntermediatelateFusionDataset, OPV2VBaseDataset,
    V2XSIMBaseDataset, DAIRV2XBaseDataset, V2XSETBaseDataset)
from opencood.data_utils.datasets import intermediate_late_fusion_dataset as ILF  # noqa: E402
from opencood.tools import inference as oc_inference           # noqa: E402
from opencood.utils import pose_utils                          # noqa: E402
from opencood.utils import eval_utils                          # noqa: E402

from src.adapters.cosdh import (cosdh_fingerprint,             # noqa: E402
                                make_faulty_base,
                                make_pose_noise_hook)
from src.adapters.runtime import FaultSpec, _InjectionLog      # noqa: E402
from src.adapters.opencood import wrapper_fingerprint          # noqa: E402
from src.fault_injectors import injector_fingerprint           # noqa: E402

_FUSION = {"late": getLateFusionDataset, "early": getEarlyFusionDataset,
           "intermediate": getIntermediateFusionDataset,
           "intermediate2stage": getIntermediate2stageFusionDataset,
           "intermediatelate": getIntermediatelateFusionDataset}
_BASE = {"opv2v": OPV2VBaseDataset, "v2xsim": V2XSIMBaseDataset,
         "dairv2x": DAIRV2XBaseDataset, "v2xset": V2XSETBaseDataset}


_AP = {}


def capture_ap():
    """``inference.main()`` computes AP but returns nothing, so wrap the
    evaluator to record it. Read-only: the original is still what computes
    and dumps, so AP/NMS remain byte-identical."""
    orig = eval_utils.eval_final_results

    def wrapped(result_stat, save_path, infer_info=None):
        out = orig(result_stat, save_path, infer_info)
        try:
            _AP["ap_30"], _AP["ap_50"], _AP["ap_70"] = (
                float(out[0]), float(out[1]), float(out[2]))
        except Exception:                                   # noqa: BLE001
            pass
        _AP["n_gt"] = int(result_stat[0.5].get("gt", -1))
        return out
    eval_utils.eval_final_results = wrapped
    oc_inference.eval_utils.eval_final_results = wrapped


def install(spec, log):
    """Patch the two interception points. Idempotent."""
    def build_dataset(dataset_cfg, visualize=False, train=True):
        fusion_name = dataset_cfg["fusion"]["core_method"]
        dataset_name = dataset_cfg["fusion"]["dataset"]
        base_cls = _BASE[dataset_name.lower()]
        if spec is not None:
            base_cls = make_faulty_base(base_cls, spec, log)
            print("[fi] base dataset wrapped -> %s" % base_cls.__name__)
        return _FUSION[fusion_name.lower()](base_cls)(
            params=dataset_cfg, visualize=visualize, train=train)

    oc_inference.build_dataset = build_dataset

    if spec is not None and spec.pose_error is not None:
        ILF.add_noise_data_dict = make_pose_noise_hook(
            pose_utils.add_noise_data_dict, spec, log)
        print("[fi] pose hook installed AFTER add_noise_data_dict "
              "(lidar_pose_clean preserved)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--fusion_method", default="intermediatelate")
    ap.add_argument("--fi_spec", default="null",
                    help="JSON FaultSpec kwargs; 'null' = UNWRAPPED (official "
                         "pipeline untouched); '{}' = wrapped, empty pipeline")
    ap.add_argument("--fi_seed", type=int, default=1234)
    ap.add_argument("--fi_out", required=True)
    ap.add_argument("--epoch", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.fi_out, exist_ok=True)
    raw = json.loads(args.fi_spec)
    spec = None if raw is None else FaultSpec(seed=args.fi_seed, **raw)
    log = None if spec is None else _InjectionLog(
        os.path.join(args.fi_out, "injection"))
    if spec is None:
        print("[fi] UNWRAPPED: official pipeline, no interception")
    else:
        os.makedirs(os.path.join(args.fi_out, "injection"), exist_ok=True)
        install(spec, log)

    capture_ap()
    sys.argv = ["inference.py", "--model_dir", args.model_dir,
                "--fusion_method", args.fusion_method,
                "--epoch", str(args.epoch)]
    rc = oc_inference.main()

    with open(os.path.join(args.fi_out, "fi_result.json"), "w") as fh:
        json.dump({**_AP,
                   "model_dir": args.model_dir, "seed": args.fi_seed,
                   "spec_json": args.fi_spec,
                   "spec": spec.as_dict() if spec else None,
                   "fusion_method": args.fusion_method,
                   "opencood_path": opencood.__file__,
                   # PROVENANCE -- must match tools/fi_inference.py EXACTLY.
                   # Omitting these is what made the cosdh gate fail closed
                   # (GATE-NO-CONTEMPORANEOUS-REF: no job_id to match on) and
                   # made cosdh's fog cells read as a MIXED-INJECTOR-VERSION.
                   **wrapper_fingerprint(),
                   **injector_fingerprint(),
                   **cosdh_fingerprint(),
                   "job_id": os.environ.get("SLURM_JOB_ID")}, fh, indent=1)
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
