"""
aggregate.py
------------
Walk the raw sweep bundles under ``results/sweep/<model>/<injector>/<tier>/``
and build the tidy results table + provenance manifest:

    results/sweep/sweep_results.csv     one row per cell, fixed 23-col schema
    results/sweep/sweep_manifest.json   grid, floors, caveats, flags

Also runs the per-cell FIRE-CHECK (injection log non-empty, magnitude matches
theory, clean logs zero injections) and the sanity gates (positive delta,
non-monotonic severity outside snow, truncated eval). A failed fire-check
marks the row's cell in ``manifest['failed_cells']`` and prints FIRE-FAIL --
it is never silently recorded as 0.00 degradation.

    .venv-hpc/bin/python tools/sweep/aggregate.py [--root results/sweep]
"""

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tools.sweep.grid import INJECTORS, MODELS, SEED, TIER_NAMES  # noqa: E402

COLUMNS = ['model', 'dataset', 'setting', 'injector', 'severity_tier',
           'severity_value', 'severity_unit', 'seed', 'clean_ap50',
           'clean_ap70', 'faulty_ap50', 'faulty_ap70', 'delta_ap50',
           'delta_ap70', 'rel_drop_ap50', 'rel_drop_ap70', 'n_injections',
           'n_frames', 'determinism_floor_ap50', 'determinism_floor_ap70',
           'extra_metric', 'job_id', 'timestamp']


# Slurm jobs that ran with the PRE-FIX adapter (from_canonical recomputed
# transformation_matrix from the clean pose, stripping shipped loc_err noise).
# Two of these were still in flight when the post-fix re-runs were submitted
# and copied cells into the SAME results path -- 558077 landed two snow cells
# into the post-fix where2comm tree. Cell provenance is therefore checked by
# job id, not by directory, because directory identity is not sufficient when
# an old job outlives the re-run's submission.
PREFIX_JOBS = {'558076', '558077', '558078',      # where2comm, pre-fix
               '558105', '558106', '558107'}      # v2xvit, pre-fix

# Primary provenance test. Cells written from 2026-08-07 carry
# `wrapper_contract_version` (see src/adapters/opencood.py); anything below
# this ran an adapter whose output semantics are known-defective.
MIN_WRAPPER_CONTRACT_VERSION = 2


def read_bundle(path):
    f = os.path.join(path, 'fi_result.json')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        r = json.load(fh)
    # Provenance, primary: the recorded contract version. Self-maintaining --
    # a future defective adapter is caught by bumping the constant, with no
    # job ids to remember.
    ver = r.get('wrapper_contract_version')
    if ver is not None and ver < MIN_WRAPPER_CONTRACT_VERSION:
        r['_prefix_contaminated'] = 'contract v%s < v%s' % (
            ver, MIN_WRAPPER_CONTRACT_VERSION)
    # Belt-and-braces: cells produced BEFORE the field existed cannot be
    # retro-tagged, so the enumerated job list still covers them.
    jid = str(r.get('job_id') or '')
    if jid in PREFIX_JOBS:
        r['_prefix_contaminated'] = 'job %s (pre-versioning)' % jid
    r['_timestamp'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(f)).isoformat(timespec='seconds')
    rows = []
    for c in glob.glob(os.path.join(path, 'injection',
                                    'injection_summary.*.csv')):
        rows += list(csv.DictReader(open(c)))
    r['_log'] = rows
    return r


def parse_detail(rows, key):
    out = []
    pat = re.compile(r'%s=(-?[\d.]+)' % re.escape(key))
    for r in rows:
        m = pat.search(r['detail'])
        if m:
            out.append(float(m.group(1)))
    return out


def parse_pts(rows):
    out = []
    for r in rows:
        m = re.search(r'pts=(\d+)->(\d+)', r['detail'])
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def fire_check(model, injector, tier_value, spec, log, nonego_denom, flags):
    """Return (ok, n_injections, extra_metric). Appends FIRE-FAIL to flags."""
    n = len(log)
    stage_rows = log

    def fail(msg):
        flags.append('FIRE-FAIL %s/%s/%s: %s' % (model, injector,
                                                 tier_value, msg))
        return False

    if n == 0:
        return fail('injection log EMPTY (silent no-op)'), 0, ''

    if injector == 'pose_error':
        sig = spec['pose_error']['sigma_xy']
        dx = [abs(v) for v in parse_detail(stage_rows, 'dx')]
        want = sig * math.sqrt(2 / math.pi)
        ok = abs(mean(dx) - want) / want < 0.15
        extra = 'mean_abs_dx=%.4f;theory=%.4f' % (mean(dx), want)
        return (True if ok else fail('mean|dx| %.4f vs %.4f' %
                                     (mean(dx), want))), n, extra
    if injector == 'latency':
        want = round(spec['latency']['mu_delay'] * 10)
        d = parse_detail(stage_rows, 'delay_frames')
        # delays are clamped at scene starts, so mean is slightly under `want`
        ok = d and max(d) == want and mean(d) > 0.8 * want
        extra = 'mean_delay_frames=%.3f;nominal=%d' % (mean(d), want)
        return (True if ok else fail('delays %s vs nominal %d' %
                                     (sorted(set(d))[:5], want))), n, extra
    if injector == 'agent_drop':
        p = spec['agent_drop']['p_drop']
        rate = n / nonego_denom if nonego_denom else float('nan')
        ok = nonego_denom and abs(rate - p) < 0.2 * p
        extra = 'p_drop=%.2f;dropped=%d;nonego_agent_frames=%s;rate=%.4f' \
                % (p, n, nonego_denom, rate)
        return (True if ok else fail('drop rate %.4f vs p %.2f' %
                                     (rate, p))), n, extra
    if injector == 'missing_modality':
        p = spec['missing_lidar']['p_drop_lidar']
        rate = n / nonego_denom if nonego_denom else float('nan')
        ok = nonego_denom and abs(rate - p) < 0.2 * p
        extra = 'p=%.2f;lidar_dropped=%d;rate=%.4f' % (p, n, rate)
        return (True if ok else fail('rate %.4f vs p %.2f' % (rate, p))), n, extra
    if injector == 'points_reduce':
        keep = {1: 0.30, 2: 0.20, 3: 0.10}[spec['points_reduce']['severity']]
        pts = parse_pts(stage_rows)
        ratios = [b / a for a, b in pts if a]
        ok = ratios and abs(mean(ratios) - keep) < 0.005
        extra = 'kept_frac=%.4f' % mean(ratios)
        return (True if ok else fail('kept %.4f vs %.2f' %
                                     (mean(ratios), keep))), n, extra
    if injector == 'lidar_fog':
        i_in = parse_detail(stage_rows, 'meanI')      # first match = input
        # detail is meanI=a->b; regex grabs `a`. Parse b explicitly:
        i_out = [float(re.search(r'meanI=[\d.]+->([\d.]+)', r['detail'])
                       .group(1)) for r in stage_rows]
        ok = mean(i_out) < mean(i_in)
        extra = 'meanI_in=%.4f;meanI_out=%.4f' % (mean(i_in), mean(i_out))
        return (True if ok else fail('intensity did not drop')), n, extra
    if injector == 'lidar_snow':
        pts = parse_pts(stage_rows)
        removed = [1.0 - b / a for a, b in pts if a]
        ok = removed and mean(removed) > 0.01
        extra = 'removed_frac=%.4f' % mean(removed)
        return (True if ok else fail('no points removed')), n, extra
    return fail('unknown injector'), n, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/sweep')
    args = ap.parse_args()

    rows, flags, missing, failed = [], [], [], []
    manifest = {
        'seed': SEED,
        'grid': {k: {'unit': u, 'tiers': dict(zip(TIER_NAMES,
                                                  [t[0] for t in tiers]))}
                 for k, (u, tiers) in INJECTORS.items()},
        'models': {m: {k: v for k, v in cfg.items()}
                   for m, cfg in MODELS.items()},
        'excluded_nodes': ['hpc-node14 (staged pathologically slow; '
                           'walled job 557245)'],
        'policy': {
            'wild_setting': 'shipped per model, unmodified; faults stack on '
                            'top; clean control measured at the same shipped '
                            'setting in the same sweep',
            'clean_control': 'unwrapped official pipeline, spec null; '
                             'wrapper-null offset measured within the '
                             'determinism floor (job 557236)',
            'scopes': {'pose_error': 'non-ego', 'latency': 'non-ego',
                       'agent_drop': 'non-ego', 'missing_modality': 'non-ego',
                       'points_reduce': 'all', 'lidar_fog': 'all',
                       'lidar_snow': 'all'},
        },
        'caveats': {
            'snow_monotonicity': 'LidarSnow point-removal is severity-'
                'inverted on CARLA clouds (sev1 removes most): the noise-'
                'floor stage dominates. removed_frac is reported per cell so '
                'the non-monotonicity is visible in the data.',
            'agent_drop_scene_counts': 'AgentDrop rows are STRATIFIED by '
                'scenario initial agent count (post max_cav truncation): one '
                'row per (severity, agent-count), with per-stratum clean '
                'reference, n_frames and n_scenes. Strata AP is accumulated '
                'in the same eval pass (per-frame tp/fp teed by scenario cav '
                'count); the overall un-stratified number is deliberately '
                'not emitted for agent_drop.',
            'where2comm': 'ungraded (no published oracle at its shipped '
                'setting); read bandwidth beside AP.',
        },
        'graded': {m: cfg['graded'] for m, cfg in MODELS.items()},
        'clean_checks': {}, 'failed_cells': [], 'flags': [],
        'wrapped_null_gate': {},
        'superseded': {
            'scope': 'AP columns only (clean_ap*, faulty_ap*, delta_*, '
                     'rel_drop_*) of v2xvit and where2comm rows produced '
                     'before the 2026-08-06 pose-composition fix.',
            'reason': 'from_canonical recomputed transformation_matrix from '
                      "params['lidar_pose'], silently discarding OpenCOOD's "
                      'shipped add_loc_noise perturbation (applied to a local '
                      'copy, never written back). Null-wrapper matrix error '
                      'with NO fault injected: 0.177 (where2comm), 0.205 '
                      '(v2xvit). CoBEVT ships loc_err=false so its error was '
                      '0.000 and its cells are UNAFFECTED.',
            'fix': 'transformation_matrix is now composed as a delta onto the '
                   'baked matrix: T_new = T_orig @ inv(A_clean) @ A_perturbed '
                   '(right-multiply, derived from T_agent->ego point-transform '
                   'convention). Invariant to cur_ego_pose_flag and to shipped '
                   'noise, since both live only inside T_orig.',
            'still_valid_from_those_runs': [
                'injection_summary.csv rows and all fire-checks',
                'lidar_snow removed_frac / scatter mechanics',
                'n_frames, n_injections, agent-count strata denominators',
                'cobevt rows in full (gate measured 0.000)'],
            'models_affected': ['v2xvit', 'where2comm'],
        },
    }

    for model, mcfg in MODELS.items():
        mroot = os.path.join(args.root, model)
        if not os.path.isdir(mroot):
            continue

        clean = read_bundle(os.path.join(mroot, 'none', 'clean'))
        clean_rep = read_bundle(os.path.join(mroot, 'none', 'clean_rep'))
        if clean is None:
            flags.append('%s: no clean control; skipping model' % model)
            continue
        c50, c70 = clean['ap_50'], clean['ap_70']
        if clean['_log']:
            flags.append('FIRE-FAIL %s/clean: clean run logged %d injections'
                         % (model, len(clean['_log'])))
        floor50 = floor70 = float('nan')
        if clean_rep is not None:
            floor50 = abs(clean_rep['ap_50'] - c50)
            floor70 = abs(clean_rep['ap_70'] - c70)
            if clean_rep['_log']:
                flags.append('FIRE-FAIL %s/clean_rep: logged injections' % model)

        # ── MANDATORY wrapped-null gate (non-skippable) ─────────────────
        # The wrapper carrying an EMPTY pipeline must reproduce the unwrapped
        # SHIPPED-setting control. Absence is itself a failure: this gate
        # exists because a missing wrapped-null cell is exactly how the
        # pose-contamination bug went unnoticed for two models.
        wnull = read_bundle(os.path.join(mroot, 'none', 'wrapped_null'))
        # COMPOSITION: a gate cell of untrusted provenance is NOT a gate. It
        # must degrade to MISSING (fail closed), never be compared. Observed
        # 2026-08-07: without this, a planted pre-fix gate cell was accepted
        # and recorded verdict=PASS -- the two guards did not compose, and the
        # contamination check only ran on injector cells.
        if wnull is not None and wnull.get('_prefix_contaminated'):
            flags.append('PREFIX-CONTAMINATED %s/none/wrapped_null: %s -- the '
                         'GATE ITSELF is untrusted; treating as MISSING'
                         % (model, wnull['_prefix_contaminated']))
            wnull = None
        # The clean control and its repeat are equally load-bearing: the gate
        # is measured against them, so their provenance must hold too.
        for nm, bnd in (('clean', clean), ('clean_rep', clean_rep)):
            if bnd is not None and bnd.get('_prefix_contaminated'):
                flags.append('PREFIX-CONTAMINATED %s/none/%s: %s -- the gate '
                             'reference is untrusted' % (model, nm,
                                                         bnd['_prefix_contaminated']))
        gate_floor = max(floor50 if floor50 == floor50 else 0.0,
                         floor70 if floor70 == floor70 else 0.0, 1e-4)
        if wnull is None:
            flags.append('GATE-MISSING %s: no none/wrapped_null cell -- the '
                         'mandatory wrapped-null gate did not run; fault '
                         'cells are UNVERIFIED against adapter round-trip '
                         'error' % model)
            manifest['wrapped_null_gate'][model] = 'MISSING'
        else:
            g50 = abs(wnull['ap_50'] - c50)
            g70 = abs(wnull['ap_70'] - c70)
            ok = g50 <= gate_floor and g70 <= gate_floor
            manifest['wrapped_null_gate'][model] = {
                'unwrapped_shipped': {'ap_50': c50, 'ap_70': c70},
                'wrapped_null': {'ap_50': wnull['ap_50'],
                                 'ap_70': wnull['ap_70']},
                'd50': g50, 'd70': g70, 'floor_used': gate_floor,
                'verdict': 'PASS' if ok else 'FAIL'}
            if wnull['_log']:
                flags.append('FIRE-FAIL %s/wrapped_null: logged %d injections '
                             '(must be zero)' % (model, len(wnull['_log'])))
            if not ok:
                flags.append('GATE-FAIL %s: wrapped-null differs from '
                             'unwrapped-shipped by d50=%.3e d70=%.3e '
                             '(floor %.1e) -- the adapter alters the model '
                             'input; every fault cell for this model is '
                             'INVALID' % (model, g50, g70, gate_floor))

        # cross-job clean checks (parts b, c)
        for part in ('b', 'c'):
            chk = read_bundle(os.path.join(mroot, 'none', 'clean_%s' % part))
            if chk is not None:
                d50 = abs(chk['ap_50'] - c50)
                d70 = abs(chk['ap_70'] - c70)
                manifest['clean_checks']['%s_%s' % (model, part)] = {
                    'ap_50': chk['ap_50'], 'ap_70': chk['ap_70'],
                    'd50_vs_control': d50, 'd70_vs_control': d70}
                if not math.isnan(floor70) and (d50 > 10 * max(floor50, 1e-6)
                                                or d70 > 10 * max(floor70, 1e-6)):
                    flags.append('%s: part-%s clean differs from control '
                                 'beyond 10x floor (d50=%.2e d70=%.2e)'
                                 % (model, part, d50, d70))

        exp_frames = mcfg['expected_frames'] or clean.get('n_frames')

        def emit(injector, tier, value, unit, bundle, n_inj, extra,
                 clean50=None, clean70=None, n_frames_override=None,
                 skip_trunc=False):
            f50, f70 = bundle['ap_50'], bundle['ap_70']
            r50 = clean50 if clean50 is not None else c50
            r70 = clean70 if clean70 is not None else c70
            rows.append({
                'model': model, 'dataset': mcfg['dataset'],
                'setting': mcfg['setting'], 'injector': injector,
                'severity_tier': tier, 'severity_value': value,
                'severity_unit': unit, 'seed': SEED,
                'clean_ap50': '%.6f' % r50, 'clean_ap70': '%.6f' % r70,
                'faulty_ap50': '%.6f' % f50, 'faulty_ap70': '%.6f' % f70,
                'delta_ap50': '%.6f' % (f50 - r50),
                'delta_ap70': '%.6f' % (f70 - r70),
                'rel_drop_ap50': '%.6f' % ((f50 - r50) / r50 if r50 else 0),
                'rel_drop_ap70': '%.6f' % ((f70 - r70) / r70 if r70 else 0),
                'n_injections': n_inj,
                'n_frames': (n_frames_override if n_frames_override is not None
                             else bundle.get('n_frames') or ''),
                'determinism_floor_ap50': '%.2e' % floor50,
                'determinism_floor_ap70': '%.2e' % floor70,
                'extra_metric': extra,
                'job_id': bundle.get('job_id') or '',
                'timestamp': bundle['_timestamp'],
            })
            if not skip_trunc and bundle.get('n_frames') and exp_frames \
                    and bundle['n_frames'] != exp_frames:
                flags.append('%s/%s/%s: n_frames %s != %s (TRUNCATED EVAL)'
                             % (model, injector, tier, bundle['n_frames'],
                                exp_frames))

        # clean-control row
        emit('none', 'clean', '', '', clean, 0, '')

        # non-ego agent-frame denominator: every pose cell logs each non-ego
        # agent exactly once per frame, so its row count IS the denominator.
        pose_mild = read_bundle(os.path.join(mroot, 'pose_error', 'mild'))
        nonego_denom = len(pose_mild['_log']) if pose_mild else None

        per_inj_deltas = {}
        for injector, (unit, tiers) in INJECTORS.items():
            for tier, (value, spec) in zip(TIER_NAMES, tiers):
                b = read_bundle(os.path.join(mroot, injector, tier))
                if b is None:
                    missing.append('%s/%s/%s' % (model, injector, tier))
                    continue
                if b.get('_prefix_contaminated'):
                    # A pre-fix job wrote into the post-fix tree (old jobs can
                    # outlive the re-run's submission). Refuse the cell rather
                    # than silently emitting a contaminated AP row.
                    flags.append('PREFIX-CONTAMINATED %s/%s/%s: written by '
                                 'job %s which ran the pre-fix adapter; cell '
                                 'REFUSED (re-run required)'
                                 % (model, injector, tier,
                                    b['_prefix_contaminated']))
                    missing.append('%s/%s/%s (prefix-contaminated)'
                                   % (model, injector, tier))
                    continue
                ok, n_inj, extra = fire_check(model, injector, value, spec,
                                              b['_log'], nonego_denom, flags)
                if not ok:
                    failed.append('%s/%s/%s' % (model, injector, tier))

                if injector == 'agent_drop':
                    # STRATIFIED reporting (approved correction): one row per
                    # (severity, scene-agent-count). Never average a p=0.5
                    # drop across 2-agent (near-binary) and 5-agent (mild)
                    # scenes. Per-stratum clean reference from the clean
                    # bundle's own strata.
                    strata = b.get('strata') or {}
                    cstrata = clean.get('strata') or {}
                    fmap = b.get('frame_stratum') or []
                    if not strata or not cstrata:
                        flags.append('%s/agent_drop/%s: strata missing from '
                                     'bundle -- driver predates the '
                                     'stratification patch' % (model, tier))
                        emit(injector, tier, value, unit, b, n_inj, extra)
                    for ncav, st in sorted(strata.items()):
                        cs = cstrata.get(ncav)
                        if cs is None:
                            flags.append('%s/agent_drop: no clean stratum '
                                         'for ncav=%s' % (model, ncav))
                            continue
                        dropped = [r for r in b['_log']
                                   if r['stage'] == 'agent_drop'
                                   and int(r['idx']) < len(fmap)
                                   and fmap[int(r['idx'])] == int(ncav)]
                        mean_dropped = (len(dropped) / st['n_frames']
                                        if st['n_frames'] else 0.0)
                        emit(injector, tier, value, unit,
                             {'ap_50': st['ap_50'], 'ap_70': st['ap_70'],
                              'n_frames': st['n_frames'],
                              'job_id': b.get('job_id'),
                              '_timestamp': b['_timestamp']},
                             len(dropped),
                             'scene_agent_count=%s,mean_dropped=%.3f,'
                             'n_scenes=%d' % (ncav, mean_dropped,
                                              st['n_scenes']),
                             clean50=cs['ap_50'], clean70=cs['ap_70'],
                             n_frames_override=st['n_frames'],
                             skip_trunc=True)
                        per_inj_deltas.setdefault(
                            'agent_drop[n=%s]' % ncav, []).append(
                            (tier, st['ap_70'] - cs['ap_70']))
                else:
                    emit(injector, tier, value, unit, b, n_inj, extra)
                    per_inj_deltas.setdefault(injector, []).append(
                        (tier, b['ap_70'] - c70))

        # sanity gates
        for injector, dl in per_inj_deltas.items():
            for tier, d in dl:
                if d > max(10 * (floor70 if not math.isnan(floor70)
                                 else 1e-4), 1e-3):
                    flags.append('%s/%s/%s: POSITIVE delta_ap70 %+0.4f '
                                 '(fault improved AP)' % (model, injector,
                                                          tier, d))
            got = dict(dl)
            if len(got) == 3 and got['mild'] < got['severe'] - max(
                    10 * (floor70 if not math.isnan(floor70) else 1e-4), 1e-3):
                note = ' (expected for snow)' if injector == 'lidar_snow' \
                    else ' (UNEXPECTED)'
                flags.append('%s/%s: mild degrades MORE than severe '
                             '(mild %+.4f vs severe %+.4f)%s'
                             % (model, injector, got['mild'], got['severe'],
                                note))

    os.makedirs(args.root, exist_ok=True)
    out_csv = os.path.join(args.root, 'sweep_results.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    manifest['failed_cells'] = failed
    manifest['missing_cells'] = missing
    manifest['flags'] = flags
    out_json = os.path.join(args.root, 'sweep_manifest.json')
    with open(out_json, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print('%d rows -> %s' % (len(rows), out_csv))
    print('%d missing cells, %d FAILED fire-checks' % (len(missing),
                                                       len(failed)))
    for f in flags:
        print('FLAG:', f)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
