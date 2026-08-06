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
    p.add_argument('--fi_condition', default=None, choices=sorted(CONDITIONS),
                   help='a named condition from CONDITIONS')
    p.add_argument('--fi_spec', default=None,
                   help="FaultSpec kwargs as JSON, e.g. "
                        "'{\"lidar_fog\": {\"severity\": 2}}'. "
                        "'null' JSON = unwrapped official control; '{}' = "
                        "null-pipeline wrapper. Takes precedence over "
                        "--fi_condition.")
    p.add_argument('--fi_out', required=True, help='per-condition output dir')
    p.add_argument('--fi_seed', type=int, default=1234,
                   help='seeds BOTH the global numpy RNG (shuffle_points) and '
                        'the injector SeedSequence')
    p.add_argument('--fi_force_wild_off', action='store_true',
                   help='force wild_setting loc_err/async off (the PoC '
                        'protocol). DEFAULT IS OFF for the sweep: each model '
                        'runs at its own shipped setting, and faults inject '
                        'on top of it.')
    p.add_argument('--fi_ns', default='opencood',
                   choices=('opencood', 'v2xvit'),
                   help='code namespace: opencood (CoBEVT/Where2comm) or '
                        'v2xvit (the V2X-ViT fork; identical basedataset, '
                        '2-arg eval signatures, no global-sort)')
    p.add_argument('--global_sort_detections', action='store_true')
    args = p.parse_args()
    if args.fi_spec is None and args.fi_condition is None:
        p.error('one of --fi_spec / --fi_condition is required')
    return args


def main():
    args = parse()
    os.makedirs(args.fi_out, exist_ok=True)
    log_dir = os.path.join(args.fi_out, 'injection')
    os.makedirs(log_dir, exist_ok=True)

    # Seed before anything touches numpy. Workers inherit this state at fork.
    np.random.seed(args.fi_seed)

    import importlib
    ns = args.fi_ns
    ocds = importlib.import_module(ns + '.data_utils.datasets')
    yaml_utils = importlib.import_module(ns + '.hypes_yaml.yaml_utils')
    eval_utils = importlib.import_module(ns + '.utils.eval_utils')

    base_name = 'IntermediateFusionDataset'
    base_cls = ocds.__all__[base_name]

    if args.fi_spec is not None:
        cond = json.loads(args.fi_spec)          # null / {} / {...} all valid
    else:
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
    # SWEEP DEFAULT (2026-08-05, per approved design): each model runs at its
    # own SHIPPED wild_setting -- faults inject on top of it, and the clean
    # control for the delta is measured at the same shipped setting in the
    # same session. The shipped values are recorded, never modified.
    # --fi_force_wild_off restores the earlier PoC protocol (both forced
    # false) for runs that need a single-fault-source guarantee.
    _orig_load_yaml = yaml_utils.load_yaml
    shipped_wild = {}

    def _load_yaml(file, opt=None):
        hypes = _orig_load_yaml(file, opt)
        ws = hypes.get('wild_setting')
        if ws is not None:
            shipped_wild.update({'loc_err': ws.get('loc_err'),
                                 'async': ws.get('async'),
                                 'xyz_std': ws.get('xyz_std'),
                                 'ryp_std': ws.get('ryp_std'),
                                 'async_overhead': ws.get('async_overhead')})
            if args.fi_force_wild_off:
                ws['loc_err'] = False
                ws['async'] = False
                print('[fi] wild_setting FORCED OFF (shipped was %s)'
                      % shipped_wild)
            else:
                print('[fi] wild_setting SHIPPED, unmodified: %s'
                      % shipped_wild)
        return hypes

    yaml_utils.load_yaml = _load_yaml

    # capture the eval denominator (n_frames must equal the test-set length
    # or the cell is truncated) and the frame -> scene-agent-count stratum
    # map for stratified AP (AgentDrop reporting: a p=0.5 drop on a 2-agent
    # scene is near-binary, on a 5-agent scene mild; averaging them into one
    # number hides that).
    oc_inf = importlib.import_module(ns + '.tools.inference')
    _orig_build = oc_inf.build_dataset
    n_frames = {}
    frame_stratum = []              # frame idx -> scenario initial cav count
    stratum_scenes = {}             # cav count -> number of scenarios

    def _build(hypes, visualize=False, train=True):
        ds = _orig_build(hypes, visualize=visualize, train=train)
        n_frames['n'] = len(ds)
        prev = 0
        for i, upper in enumerate(ds.len_record):
            ncav = len(ds.scenario_database[i])
            stratum_scenes[ncav] = stratum_scenes.get(ncav, 0) + 1
            frame_stratum.extend([ncav] * (upper - prev))
            prev = upper
        assert len(frame_stratum) == len(ds), 'stratum map misaligned'
        return ds

    oc_inf.build_dataset = _build

    # tee the per-frame tp/fp accumulation into per-stratum stats. The eval
    # loop calls caluclate_tp_fp exactly 3x per frame (iou 0.3/0.5/0.7),
    # batch=1, shuffle=False, no skip path -- so a per-iou call counter IS
    # the frame index. The global result_stat sees exactly the calls it
    # always saw; strata get their own accumulation from the same tensors
    # (pure recomputation, no RNG), so overall AP is untouched.
    _orig_tpfp = eval_utils.caluclate_tp_fp
    strata_stat = {}                # ncav -> result_stat-shaped dict
    call_counter = {}

    def _tpfp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
        out = _orig_tpfp(det_boxes, det_score, gt_boxes, result_stat,
                         iou_thresh)
        k = call_counter.get(iou_thresh, 0)
        call_counter[iou_thresh] = k + 1
        if k < len(frame_stratum):
            ncav = frame_stratum[k]
            st = strata_stat.setdefault(ncav, {
                0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}})
            _orig_tpfp(det_boxes, det_score, gt_boxes, st, iou_thresh)
        return out

    eval_utils.caluclate_tp_fp = _tpfp

    # ── capture AP at full precision, into our own dir ──────────────────
    # `eval_final_results` prints only %.2f and writes eval.yaml into
    # opt.model_dir, where per-condition runs would overwrite each other.
    _orig_eval = eval_utils.eval_final_results
    captured = {}

    def _eval(result_stat, save_path, *rest):
        out = _orig_eval(result_stat, args.fi_out, *rest)
        try:
            import yaml
            with open(os.path.join(args.fi_out, 'eval.yaml')) as fh:
                captured.update(yaml.safe_load(fh))
        except Exception as exc:                          # pragma: no cover
            print('[fi] could not re-read eval.yaml: %r' % exc)
        print('[fi] FULL PRECISION  ap_50=%r  ap_70=%r'
              % (captured.get('ap_50'), captured.get('ap_70')))
        strata = {}
        for ncav, st in sorted(strata_stat.items()):
            try:
                gs = rest if rest else ()
                a50 = eval_utils.calculate_ap(st, 0.50, *gs)[0]
                a70 = eval_utils.calculate_ap(st, 0.70, *gs)[0]
            except Exception as exc:              # pragma: no cover
                print('[fi] stratum %d AP failed: %r' % (ncav, exc))
                continue
            strata[str(ncav)] = {
                'ap_50': a50, 'ap_70': a70,
                'n_frames': frame_stratum.count(ncav),
                'n_scenes': stratum_scenes.get(ncav, 0)}
        if strata:
            print('[fi] strata (by scene agent count): %s'
                  % json.dumps(strata, sort_keys=True))
        with open(os.path.join(args.fi_out, 'fi_result.json'), 'w') as fh:
            json.dump({'condition': args.fi_condition,
                       'spec_json': args.fi_spec,
                       'seed': args.fi_seed,
                       'spec': spec.as_dict() if spec else None,
                       'model_dir': args.model_dir,
                       'wild_setting_shipped': shipped_wild,
                       'wild_forced_off': bool(args.fi_force_wild_off),
                       'n_frames': n_frames.get('n'),
                       'job_id': os.environ.get('SLURM_JOB_ID'),
                       'ap_30': captured.get('ap30'),
                       'ap_50': captured.get('ap_50'),
                       'ap_70': captured.get('ap_70'),
                       'strata': strata,
                       'frame_stratum': frame_stratum}, fh, indent=2)
        return out

    eval_utils.eval_final_results = _eval

    sys.argv = ['inference.py',
                '--model_dir', args.model_dir,
                '--fusion_method', args.fusion_method]
    if args.global_sort_detections and ns == 'opencood':
        sys.argv.append('--global_sort_detections')

    oc_inf.main()


if __name__ == '__main__':
    main()
