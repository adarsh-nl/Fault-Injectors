"""
fi_inference.py
---------------
Run OpenCOOD inference with fault injection.

This is the ONLY file in the project that imports OpenCOOD. It installs the
faulty dataset class into ``opencood.data_utils.datasets.__all__`` and then
calls the **unmodified official** ``inference.main()``. ``build_dataset`` reads
``__all__`` at call time, so the swap takes effect while every line from
``__getitem__`` downward -- preprocessing, model, NMS, ``eval_final_results`` --
stays byte-identical to the code that produced the verified baselines. Forking
``inference.py`` would put AP computation on a branch that can silently drift.

Why the global numpy seed
-------------------------
``get_item_single_car`` calls ``shuffle_points``, which is
``np.random.permutation`` on the **global** RNG (``pcd_utils.py:92``). Point
order decides which points survive per-pillar truncation
(``max_points_per_voxel: 32``), so AP at full precision depends on that RNG, and
nothing in the eval path seeds it. Unseeded, two identical runs differ in the
last digits -- so "reproduce 0.85/0.66 exactly" is not a usable gate. Seeding
here makes it one: a null-pipeline run must match a *freshly measured* official
control at full precision, in the same session, same seed, same worker count.

``PoseErrorInjector`` draws from a local ``default_rng``, so it does not disturb
that global stream -- which is what makes the clean/faulty delta attributable.

Usage
-----
    python tools/fi_inference.py \
        --model_dir ~/opencood-eval/cobevt \
        --fusion_method intermediate \
        --fi_condition clean \
        --fi_out $SCRATCH/out/clean
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import FaultSpec, make_faulty_dataset   # noqa: E402

# The conditions this PoC runs. `clean` is a NULL pipeline that still traverses
# the whole adapter round trip, so Gate 2 tests the adapter and not a bypass.
CONDITIONS = {
    'official': None,                                     # no wrapper at all
    'clean':    dict(),                                   # null spec
    'pose_sev2': dict(pose_error={'sigma_xy': 0.2, 'sigma_heading': 0.2}),
}


def parse():
    p = argparse.ArgumentParser(description='OpenCOOD inference + faults')
    p.add_argument('--model_dir', required=True)
    p.add_argument('--fusion_method', default='intermediate')
    p.add_argument('--fi_condition', required=True, choices=sorted(CONDITIONS))
    p.add_argument('--fi_out', required=True, help='per-condition output dir')
    p.add_argument('--fi_seed', type=int, default=1234,
                   help='seeds BOTH the global numpy RNG (shuffle_points) and '
                        'the injector SeedSequence')
    p.add_argument('--global_sort_detections', action='store_true')
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.fi_out, exist_ok=True)
    log_dir = os.path.join(args.fi_out, 'injection')
    os.makedirs(log_dir, exist_ok=True)

    # Seed before anything touches numpy. Workers inherit this state at fork.
    np.random.seed(args.fi_seed)

    import opencood.data_utils.datasets as ocds
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.utils import eval_utils

    base_name = 'IntermediateFusionDataset'
    base_cls = ocds.__all__[base_name]

    cond = CONDITIONS[args.fi_condition]
    if cond is not None:
        spec = FaultSpec(seed=args.fi_seed, log_dir=log_dir, **cond)
        ocds.__all__[base_name] = make_faulty_dataset(base_cls, spec)
        print('[fi] wrapped %s -> %s | spec=%s'
              % (base_name, ocds.__all__[base_name].__name__,
                 json.dumps(spec.as_dict(), sort_keys=True)))
    else:
        spec = None
        print('[fi] UNWRAPPED official dataset (control run)')

    # ── wild_setting policy ─────────────────────────────────────────────
    # OpenCOOD's own loc_err/async are forced off whenever our injectors run.
    # `add_loc_noise` reseeds the GLOBAL numpy RNG with a fixed seed on every
    # call, so its noise is the same draw for every agent and every frame --
    # stacking it under our injectors gives a delta that is neither
    # attributable nor a well-behaved nuisance term. The clean control is
    # measured at the same forced setting so the two are comparable.
    _orig_load_yaml = yaml_utils.load_yaml

    def _load_yaml(file, opt=None):
        hypes = _orig_load_yaml(file, opt)
        ws = hypes.get('wild_setting')
        if ws is not None:
            was = {'loc_err': ws.get('loc_err'), 'async': ws.get('async')}
            ws['loc_err'] = False
            ws['async'] = False
            assert ws['loc_err'] is False and ws['async'] is False
            print('[fi] wild_setting forced off (was %s)' % was)
        return hypes

    yaml_utils.load_yaml = _load_yaml

    # ── capture AP at full precision, into our own dir ──────────────────
    # `eval_final_results` prints only %.2f and writes eval.yaml into
    # opt.model_dir, where per-condition runs would overwrite each other.
    _orig_eval = eval_utils.eval_final_results
    captured = {}

    def _eval(result_stat, save_path, global_sort_detections):
        out = _orig_eval(result_stat, args.fi_out, global_sort_detections)
        try:
            import yaml
            with open(os.path.join(args.fi_out, 'eval.yaml')) as fh:
                captured.update(yaml.safe_load(fh))
        except Exception as exc:                          # pragma: no cover
            print('[fi] could not re-read eval.yaml: %r' % exc)
        print('[fi] FULL PRECISION  ap_50=%r  ap_70=%r'
              % (captured.get('ap_50'), captured.get('ap_70')))
        with open(os.path.join(args.fi_out, 'fi_result.json'), 'w') as fh:
            json.dump({'condition': args.fi_condition,
                       'seed': args.fi_seed,
                       'spec': spec.as_dict() if spec else None,
                       'ap_30': captured.get('ap30'),
                       'ap_50': captured.get('ap_50'),
                       'ap_70': captured.get('ap_70')}, fh, indent=2)
        return out

    eval_utils.eval_final_results = _eval

    sys.argv = ['inference.py',
                '--model_dir', args.model_dir,
                '--fusion_method', args.fusion_method]
    if args.global_sort_detections:
        sys.argv.append('--global_sort_detections')

    from opencood.tools.inference import main as oc_main
    oc_main()


if __name__ == '__main__':
    main()
