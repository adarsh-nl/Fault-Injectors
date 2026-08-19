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

    na = []          # NOT-APPLICABLE cells (see NOT_APPLICABLE)
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
            'snow_monotonicity': 'TWO SEPARATE FACTS, corrected 2026-08-10 '
                '-- the earlier text implied the AP non-monotonicity was '
                'general, which the completed grid contradicts. '
                '(1) THE REMOVAL-FRACTION INVERSION IS UNIVERSAL AND '
                'MODEL-INDEPENDENT. LidarSnow strips MOST points at sev1: '
                'measured mean removed fraction on V2XSet is 0.403 / 0.326 / '
                '0.291 for mild / moderate / severe, agreeing to three '
                'decimals across all three models (cobevt, where2comm, '
                'v2xvit) -- as it must, since the corruption plane runs on '
                'the CooperativeSample before any model sees it. The same '
                'ordering holds on OPV2V at a higher level (0.578 / 0.516 / '
                '0.489; denser removal, sparser clouds). The mechanism is '
                'that the noise-floor stage dominates the scatter stage at '
                'low severity. '
                '(2) WHETHER IT SURFACES IN AP IS MODEL-DEPENDENT, and '
                'mostly it does NOT. Only where2comm on V2XSet inverts in '
                'AP@0.7 (mild 0.2228 < moderate 0.2342 > severe 0.2075) -- '
                'the ONLY AP inversion in the entire 78-cell grid. cobevt '
                '(0.2664 / 0.2566 / 0.2123) and v2xvit (0.1916 / 0.1729 / '
                '0.1345) stay monotone on V2XSet, and where2comm on OPV2V '
                'was monotone too (0.2935 / 0.2797 / 0.2337). So the scatter '
                'stage dominates AP degradation for most models even where '
                'mild strips more points. '
                'HYPOTHESIS, NOT ESTABLISHED CAUSE: where2comm may be the '
                'one model to show it because its confidence-based selection '
                'depends directly on point density, so a removal-heavy mild '
                'setting degrades its request/confidence maps more than the '
                'added scatter does. This has NOT been tested -- doing so '
                'would need the bandwidth/selection columns read alongside '
                'AP across the three snow tiers. removed_frac is reported '
                'per cell so both facts are checkable in the data.',
            'agent_drop_scene_counts': 'AgentDrop rows are STRATIFIED by '
                'scenario initial agent count (post max_cav truncation): one '
                'row per (severity, agent-count), with per-stratum clean '
                'reference, n_frames and n_scenes. Strata AP is accumulated '
                'in the same eval pass (per-frame tp/fp teed by scenario cav '
                'count); the overall un-stratified number is deliberately '
                'not emitted for agent_drop.',
            'where2comm_provenance': 'THIRD-PARTY RETRAINING, NOT AN '
                'AUTHOR RELEASE. The checkpoint is md5-identical '
                '(4beff417ffe6c62d76c88acaff63d32c) to the one inside '
                'v2xset_checkpoints/point_pillar_where2comm_v2xset.zip, i.e. '
                'it is Where2comm trained on V2XSET. V2XSet is the V2X-ViT '
                "authors' dataset; the Where2comm paper evaluates on OPV2V, "
                'V2X-Sim, DAIR-V2X and CoPerception-UAVs, and the official '
                'Where2comm repo releases NO checkpoints at all (verified '
                '2026-08-08: its README has no model zoo). The archive sits '
                'in a V2XSet baseline collection beside cobevt_lidar.zip, so '
                'it is a re-training published as a V2XSet baseline by a '
                'third party. CONSEQUENCE: where2comm is ungraded because no '
                'number attributable to the Where2comm authors exists for '
                'this model/dataset pair -- the reason is PROVENANCE, not a '
                'missing entry in a published table. Read bandwidth beside '
                'AP.',
            'where2comm_modality': 'LiDAR only, and that is faithful to the '
                'paper. Verified from the paper: the camera and LiDAR tracks '
                'are PER-DATASET, not fused, and the LiDAR detector follows '
                'PointPillar -- which is exactly this checkpoint '
                '(point_pillar_where2comm, SpVoxelPreprocessor, '
                'VoxelPostprocessor, zero camera weights in 148 params). '
                "Evaluating it on LiDAR reproduces the paper's own LiDAR "
                'track; it is not a modality-ablated variant of a fused '
                'model.',
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
        'cora_notes': {
            'purpose': 'FOR THE WRITEUP, NOT FOR ACTION.',
            'seed_pair_is_not_hardware_matched': 'The two CoRA full runs '
                '(559639 seed 2026, 559640 seed 2029) landed on DIFFERENT GPU '
                'SKUs -- 559639 on an L40S (hpc-node12), 559640 on an L40 '
                '(hpc-node08). Both are permitted by '
                '--constraint=a40|a100|l40|l40s. The pair was launched to give '
                'a SEED-VARIANCE datapoint, so that the comparison against the '
                'published 0.9341/0.8658 is interpretable rather than '
                'ambiguous at n=1. Because the SKUs differ, the AP spread '
                'between the two seeds carries a HARDWARE component alongside '
                'the seed component and MUST NOT be reported as pure seed '
                'variance. Scale of the hardware term, measured elsewhere in '
                'this project: the cross-SKU determinism floor is ~4e-5 AP '
                '(v2xvit part-a vs part-b), i.e. three orders below any fault '
                'delta -- so it is very unlikely to dominate a seed spread, '
                'but it is not zero and it was not controlled.',
        },
        'gt_union_confound': {
            'status': 'MEASURED, NOT YET CORRECTED FOR. Affects agent_drop '
                      'rows only.',
            'finding': 'GT is assembled as the UNION over CAVs: both vanilla '
                'OpenCOOD (IntermediateFusionDataset accumulates '
                'object_bbx_center over all in-range CAVs, then de-duplicates) '
                'and CoSDH build it the same way. So agent_drop removes GT '
                'BOXES ALONG WITH THE AGENT -- it changes the EVALUATION '
                'TARGET, not only the model input.',
            'measured': 'Dataset-level, 40 frames, CoBEVT config on V2XSet '
                'test: GT totals 832 (clean) -> 818 / 790 / 769 at '
                'p_drop = 0.25 / 0.50 / 0.75, i.e. -1.68% / -5.05% / -7.57%. '
                'Monotone in p; the majority of frames shrink at p>=0.50.',
            'why_it_biases': 'Recall = TP/GT. A smaller GT inflates recall '
                'for the same detections, so the measured degradation is '
                'biased TOWARD LOOKING MILDER than it is. agent_drop is '
                'currently reported as the LEAST damaging fault in the grid, '
                'so part of that gap is an artifact.',
            'affected': ['Fig 4 entirely (stratified agent-drop panel; the '
                         'per-stratum clean references are computed at the '
                         "stratum's natural agent count while the faulted "
                         'bars have had agents removed, so reference and bar '
                         'are not evaluated against the same GT set)',
                         'Fig 3, agent_drop row and its position in the '
                         'ordering'],
            'not_affected': ['Fig 2 (agent_drop is already a hatched '
                             'not-poolable band there)',
                             'pose_error, latency, points_reduce, lidar_fog, '
                             'lidar_snow -- all leave the agent SET intact',
                             'missing_modality -- it empties a cloud but '
                             'KEEPS the agent in the dict, so on the OpenCOOD '
                             'path that CAV still contributes its GT objects. '
                             'Its denominator is stable. This is a genuine '
                             'asymmetry between the two "lost collaborator" '
                             'faults, which the figures currently present as '
                             'a pair.'],
            'limits': 'n_gt is NOT recorded in the OpenCOOD result bundles '
                '(only n_frames), and the 4MB eval.yaml dumps carry only '
                'ap30/ap_50/ap_70 and the PR curves -- so the per-cell '
                'shrinkage CANNOT be quantified from data already on disk. '
                'The figure above is a 40-frame dataset-level sample on one '
                "model's config on V2XSet: it establishes the mechanism and "
                'rough scale, NOT a correction factor. Recording n_gt per '
                'cell is a prerequisite for any correction.',
        },
        'cosdh_notes': {
            'purpose': 'CoSDH protocol facts. These change how its numbers may '
                       'be read and are recorded here rather than only in '
                       'src/adapters/cosdh.py.',
            'not_gradeable_against_published': 'The released checkpoints were '
                'RETRAINED and, per the release README, "differ slightly from '
                'those reported in the paper". CoSDH is therefore NOT '
                'gradeable against the published 96.83 / 92.99 the way CoBEVT '
                'and V2X-ViT are graded. Measured clean baseline (gate 1, job '
                '561691, 2170 frames, 32244 gt): AP@0.5 = 0.9660 (-0.0023 vs '
                'paper), AP@0.7 = 0.9289 (-0.0010). It IS an author release, '
                'which is stronger evidence than where2comm\'s third-party '
                'retrain, but the published number is not its oracle.',
            'shipped_setting_is_perfect': 'CoSDH ships '
                'noise_setting: {add_noise: false} -- there is no loc_err/'
                'async wild_setting at all. It is effectively PERFECT-shipped '
                'and GROUPS WITH COBEVT, not with the two Noisy models '
                '(v2xvit, where2comm).',
            'injected_sigma_is_total_not_increment': 'Because the shipped '
                'pose noise is zero, an injected pose sigma is the TOTAL pose '
                'error. For v2xvit and where2comm the same sigma stacks on '
                'top of a shipped loc_err perturbation, so equal sigma does '
                'NOT mean equal total error across models.',
            'agent_drop_and_missing_modality_are_dual_channel': 'CoSDH is '
                'intermediate-LATE hybrid: collaborators contribute a feature '
                'channel AND a detection channel (non-ego detections are '
                'thresholded at confidence_threshold 0.3 and scaled by '
                'confidence_beta 0.9, then fused). agent_drop and '
                'missing_modality remove an agent from BOTH channels at once, '
                'a STRICTLY LARGER INTERVENTION than on the three '
                'pure-intermediate models. Those two rows are not directly '
                'comparable to the others without this footnote.',
            'provenance_stamping_is_partial': 'TWO DIFFERENT SCOPES ARE IN '
                'PLAY HERE AND BOTH ARE TRUE -- do not collapse them. '
                '(a) MIXED-VERSION scope = 3 cells. lidar_fog is the only '
                'injector for which the other three models ARE stamped '
                '(bdc5c3f7 x9) while CoSDH is not, so cosdh/lidar_fog/'
                '{mild,moderate,severe} are the only cells that create a '
                'mixed-version COMPARISON hazard. That is what the '
                'MIXED-INJECTOR-VERSION flag reports and it is correctly '
                'scoped at 3. '
                '(b) UNSTAMPED scope = 20 cells. Measured 2026-08-18 across '
                'all 20 bundles: EVERY pre-existing CoSDH cell lacks job_id, '
                'wrapper_contract_version, injector_sha256, '
                'injector_files_sha256 and injector_contract_version. For '
                'the other six injectors this does not show up as "mixed" '
                'only because the other three models are unstamped there '
                'too -- uniform ignorance, not uniform provenance. '
                'The 3-cell number answers "where can I not pool cells?"; '
                'the 20-cell number answers "which CoSDH cells can I trace '
                'to a job?". '
                'tools/cosdh_inference.py only began writing those fields on '
                '2026-08-18; the cells were produced 2026-08-13. '
                'WHAT THE CELLS DO CARRY, on all 20: cosdh_adapter_sha256 = '
                '3f7230d4868c2b34 and cosdh_contract_version = 1, identical '
                'on every cell and identical to the current '
                'src/adapters/cosdh.py (sha256 3f7230d4868c2b34..., stored '
                'truncated to 16 chars). src/adapters/cosdh.py is the file '
                'that wires all six non-pose injectors and the pose hook, so '
                'the FAULT-DETERMINING code is digest-confirmed byte-'
                'identical to the stamped version for every cell. What is '
                'unconfirmed by digest is the shared src/fault_injectors '
                'code underneath it. '
                'REMEDIATION AND ITS LIMIT: the gate re-run (none/clean + '
                'none/wrapped_null, one job, contemporaneous by construction) '
                'restamps TWO cells. The other 18 were deliberately NOT '
                're-run -- ~30 GPU-hours to change a provenance field and no '
                'AP. So 18 of 20 cells remain job_id-less after remediation. '
                'THEIR LINEAGE IS NEVERTHELESS RECORDED, NOT INFERENTIAL -- '
                'this corrects an earlier statement here that called it '
                'inferential. Every bundle carries model_dir = '
                '/local/<SLURM_JOB_ID>/model, because the checkpoint is '
                'staged into per-job node-local scratch, so the job id is '
                'recoverable from a field that was always written. Measured '
                '2026-08-18, all 20 cells resolve: job 562333 -> 12 cells '
                '(latency x3, points_reduce x3, pose_error x3, none/clean, '
                'none/clean_rep, none/wrapped_null); 562334 -> 4 '
                '(lidar_fog x3, none/clean_b); 562335 -> 3 (lidar_snow '
                'mild+moderate, none/clean_c); 565554 -> 1 '
                '(lidar_snow/severe). Three of the four sweep jobs carried '
                'their own clean control; 565554 did not. '
                'CONSEQUENCE FOR THE GATE, stated plainly: none/clean and '
                'none/wrapped_null WERE ALREADY CONTEMPORANEOUS -- both come '
                'from job 562333. The gate did not fail because the cells '
                'were produced at different times; it failed because it '
                'reads the job_id FIELD, which the driver did not write, '
                'while the lineage sat in model_dir unread. The re-run is '
                'therefore a re-stamp of something already true, not the '
                'establishment of a new fact, and it doubles as an '
                'independent determinism check. model_dir was NOT '
                'retro-written into job_id: backfilling a provenance field '
                'by inference is the failure mode this whole census exists '
                'to prevent. Treat this the way V2X-ViT lidar_snow/mild is '
                'treated -- a known, bounded provenance gap that is stated '
                'rather than silently closed -- but note it is 18 cells here '
                'versus 1 there. '
                'PRE-REGISTERED EXPECTATION FOR THE RE-RUN, stated before it '
                'landed: the superseded pre-stamp pair (preserved, not '
                'deleted, at results/sweep/cosdh_PRESTAMP_gate_pair/ -- not a '
                'MODELS key, so inert to the aggregator) already gives '
                'ap_50 = ap_70 delta of EXACTLY 0.0 between clean and '
                'wrapped_null (both 0.966028710 / 0.928910153, n_gt 32244). '
                'The gate was therefore never failing on a real wrapper '
                'delta -- only on the absence of a same-job_id reference. '
                'The re-run should reproduce 0.0 / 0.0 and clean AP '
                'identical to 9 decimal places. If the re-run\'s clean AP '
                'MOVES, that is not a gate result, it is evidence of '
                'run-to-run nondeterminism in the CoSDH path and must be '
                'reported as such rather than absorbed.',
            'env_not_byte_identical': 'CoSDH runs in the separate `cosdh` '
                'conda env, which is NOT byte-identical to the '
                '`opencood-official` env that produced the other three '
                "models' results: numpy 1.21.5 vs 1.21.6, PIL 9.4.0 vs 9.5.0 "
                '(torch 1.12.1+cu113 and spconv 2.3.6 match). Internally '
                'consistent for CoSDH\'s own cells, but a cross-model '
                'comparability caveat. opencood-official was deliberately NOT '
                'mutated -- it produced all previously reported results.',
        },
        'v2xvit_notes': {
            'purpose': 'FOR THE WRITEUP, NOT FOR ACTION. Both items below are '
                       'recorded so the provenance story is complete; neither '
                       'invalidates a cell and neither requires a re-run.',
            'snow_mild_unstamped': 'results/sweep/v2xvit/lidar_snow/mild was '
                'written by job 558721, which predates wrapper-contract-'
                'version stamping, so the bundle carries no '
                'wrapper_contract_version field. It is clean by JOB-ID '
                'LINEAGE only (558721 belongs to the post-fix vit2 re-run '
                'set, not to the quarantined prefix jobs 558077/558106). It '
                'is the one v2xvit cell the contract-version guard cannot '
                'positively verify. The other two snow cells (moderate, '
                'severe) come from 559219 and are stamped version 2.',
            'determinism_floor_is_per_node': 'The part-b clean control '
                'differs from the part-a clean control by d50=4.32e-05, '
                'd70=7.74e-05 -- above the 1e-4-scaled gate floor derived '
                'from the part-a clean/clean_rep pair, which is why the '
                'aggregator flags it. Cause: parts a and b landed on '
                'DIFFERENT GPU SKUs under --constraint=a40|a100|l40|l40s. So '
                'the determinism floor is PER-NODE, not global: within one '
                'node the repeat is bit-exact (d=0.0), across SKUs it is '
                '~4e-5. The wrapped-null gate is unaffected because clean, '
                'clean_rep and wrapped_null all live in part a and therefore '
                'ran on one node. ~4e-5 is three orders below any fault '
                'delta, so no AP comparison is affected; the lesson is that '
                'a single SKU must be pinned across all parts of a model, '
                'which is what the where2comm V2XSet re-run does '
                '(--constraint=l40).',
        },
        'superseded_domain_pairing': {
            'model': 'where2comm',
            'moved_to': 'results/sweep/where2comm_SUPERSEDED_opv2v_eval/',
            'scope': 'ALL 26 cells of the previous where2comm sweep -- every '
                     'AP number, every delta, every rel_drop.',
            'reason': 'The checkpoint is V2XSET-trained but was evaluated on '
                      'OPV2V test: a DOMAIN MISMATCH, so its clean control '
                      'and every degradation measured against it describe a '
                      'cross-domain model, not the model as trained. The '
                      "config's root_dir: /data/opv2x/train is the author's "
                      'shorthand and was mistaken for evidence of OPV2V; the '
                      'md5 match against point_pillar_where2comm_v2xset.zip '
                      'is the decisive evidence and it says V2XSet.',
            'fix': 'where2comm re-paired with its training domain, '
                   'v2xset/test (2834 frames), and the full grid re-run '
                   'including the mandatory none/wrapped_null gate cell.',
            'still_valid_from_those_runs': [
                'injection_summary.csv rows and all fire-checks -- the '
                'injectors fired correctly, they simply fired on the wrong '
                'evaluation domain',
                'lidar_snow removed_frac / scatter mechanics (a property of '
                'the point clouds and the injector, not of the checkpoint)',
                'the wrapper round-trip evidence',
            ],
            'not_valid': ['every AP, delta and rel_drop column'],
            'aggregator_visibility': 'NONE. aggregate.py discovers cells by '
                'iterating MODELS.keys() and joining <root>/<model>, so a '
                'directory not named by a MODELS key is unreachable. The '
                'superseded tree is therefore inert, not merely ignored.',
        },
    }

    # ── injector-version census (mixed-version grid must be VISIBLE) ────
    # Cells written before 2026-08-10 carry NO injector fingerprint: the field
    # did not exist and they CANNOT be retro-stamped, because the hash is of
    # the source tree as it ran and that tree is gone. They are identified by
    # the ABSENCE of the field, recorded as 'unstamped (pre-2026-08-10)'. An
    # unstamped cell is only as trustworthy as its job_id lineage.
    #
    # Keyed PER-INJECTOR, not on the combined hash. The combined hash covers
    # the whole injector directory, so the lidar_fog intensity-scale fix moved
    # it for every cell -- including snow cells whose own source never
    # changed. Flagging those as incomparable would be false. What matters is
    # whether a SINGLE injector has more than one version across the grid.
    versions = {}
    for _m in MODELS:
        for _f in sorted(glob.glob(os.path.join(args.root, _m, '*', '*',
                                                'fi_result.json'))):
            with open(_f) as _fh:
                _d = json.load(_fh)
            _cell = os.path.relpath(os.path.dirname(_f),
                                    os.path.join(args.root, _m))
            _inj = _cell.split(os.sep)[0]
            if _inj == 'none':
                # clean controls apply NO corruption, so no injector version
                # can apply to them; including them produced a spurious
                # "mixed" flag purely from the clean_b cells of the fog re-run.
                continue
            _per = _d.get('injector_files_sha256') or {}
            _key = _per.get(_inj) or ('unstamped (pre-2026-08-10)'
                                      if 'injector_sha256' not in _d
                                      else 'stamped, injector not in manifest')
            versions.setdefault(_inj, {}).setdefault(_key, []).append(
                '%s/%s' % (_m, _cell))
    manifest['injector_versions'] = {
        inj: {k: {'n_cells': len(v), 'cells': sorted(v)}
              for k, v in ks.items()} for inj, ks in sorted(versions.items())}
    for _inj, ks in sorted(versions.items()):
        if len(ks) > 1:
            flags.append('MIXED-INJECTOR-VERSION %s: %d distinct source '
                         'versions for THIS injector (%s). Its cells were not '
                         'all produced by the same corruption code and must '
                         'not be pooled or compared across versions.'
                         % (_inj, len(ks),
                            ', '.join('%s x%d' % (k, len(v))
                                      for k, v in sorted(ks.items()))))

    # ── NOT-APPLICABLE cells ────────────────────────────────────────────
    # Distinct from MISSING (not yet run) and from a 0.0 AP result. A
    # not-applicable cell means the model's released inference path CANNOT
    # EXECUTE under that fault -- there is no AP to report, degraded or
    # otherwise. Emitting 0.0 would be wrong twice over: it invents a
    # measurement, and it would enter the robustness ordering as "maximally
    # degraded" when the truth is "not measurable". These are excluded from
    # every ranking, exactly as the hatched bands are in Figs 2 and 3.
    NOT_APPLICABLE = {
        ('cosdh', 'agent_drop'): (
            "NOT a fusion failure -- unlike cosdh/missing_modality, and the "
            "two must not be read as one finding. WHAT IS MEASURED: the cell "
            "dies in GROUND-TRUTH ASSEMBLY at "
            "base_postprocessor.py:110, `torch.from_numpy(gt_box3d_np).to("
            "device=gt_box3d_list[0].device)`, with IndexError: index 0 is "
            "out of bounds for dimension 0 with size 0 -- i.e. the stacked GT "
            "tensor has ZERO ROWS. This is BEFORE AP is computed. The loop "
            "runs exactly ONCE, over the ego, because the caller passes "
            "data_dict_gt['ego'] alone "
            "(intermediate_late_fusion_dataset.py:647), so the list is empty "
            "because its single entry was empty, not because the loop never "
            "ran. record_len is CONSISTENT here (measured: record_len=1 with "
            "exactly 1 agent producing features), which is why this is not "
            "the missing_modality mechanism. "
            "WHAT IS NOT ESTABLISHED -- stated plainly: THE MECHANISM PRODUCING "
            "THE EMPTY TENSOR IS NOT ESTABLISHED. This cell is reported as "
            "not-applicable on the basis of an OBSERVED failure, not an "
            "explained one. Three "
            "dataset-level hypotheses were measured and REFUTED -- a "
            "record_len mismatch; ego-alone frames yielding empty GT "
            "(ego-alone frames are common, 14/29/39 of the first 60 at "
            "p=0.25/0.50/0.75, but GT survives at 18-20 boxes); and empty GT "
            "at all (no frame in the first 60 yields zero GT; the minimum is "
            "9). Decisively, THE FRAME THE SWEEP DIED ON (index 3) had one "
            "non-ego agent still present and GT of 20, IDENTICAL TO CLEAN -- "
            "at dataset level it is indistinguishable from an unfaulted "
            "frame. The cause therefore lies BETWEEN the dataset and the "
            "inference loop (collation, the model forward, or post_process, "
            "which runs before generate_gt_bbx), and settling it would "
            "require reproducing the crash under the real inference path. "
            "That was not done: the grid consequence is identical either way. "
            "This is a property of the RELEASED EVALUATION PATH, not of the "
            "fusion architecture, and it is a WEAKER claim than the "
            "missing_modality finding. The paper does not test collaborator "
            "dropout and does not claim robustness to it."),
        ('cosdh', 'missing_modality'): (
            "Same root cause as cosdh/agent_drop. A retained agent with a "
            "ZERO-POINT cloud produces no voxels, so record_len over-counts "
            "the agents that produced features (RuntimeError: Expected size "
            "for first two dimensions of batch2 tensor to be: [1, 3] but got: "
            "[2, 3] at fusion_in_one.py:507). Fails at every severity within "
            "the first few frames. Property of the released inference path, "
            "not of the method."),
    }
    manifest['not_applicable'] = {
        '%s/%s' % k: v for k, v in NOT_APPLICABLE.items()}

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

        # ── the reference MUST be CONTEMPORANEOUS with wrapped_null ───────
        # Fixed 2026-08-10. The gate previously compared wrapped_null against
        # none/clean regardless of provenance, while deriving its floor from a
        # WITHIN-session repeat (clean vs clean_rep). For cobevt those two
        # bundles came from different jobs two weeks apart (clean=557377,
        # wrapped_null=559457) and the gate reported FAIL at d50=1.414e-04.
        # That was cross-session drift, not adapter error, and the decisive
        # measurement is that the two UNWRAPPED runs differ by MORE than the
        # delta blamed on the wrapper: 557377-clean vs 559457-unwrapped is
        # d50=1.60e-04 d70=1.30e-04, with no wrapper involved anywhere. A
        # wrapper cannot cause a delta smaller than the wrapper-free drift.
        #
        # The fix is to require the reference to come from the SAME JOB as
        # wrapped_null -- NOT to widen the floor for cross-job comparisons,
        # which would weaken the gate on every model to accommodate one case.
        # Same job implies same node, same session, same staged data, so the
        # comparison isolates the wrapper and nothing else.
        #
        # Preference order:
        #   1. none/clean, if it carries the same job_id as wrapped_null
        #      (true for where2comm and v2xvit: both live in part a)
        #   2. results/null_gate/<model>/unwrapped, if SAME job_id -- the
        #      dedicated gate job runs both passes back to back
        #   3. nothing -> MISSING. FAIL CLOSED.
        # A null_gate bundle from a DIFFERENT job is rejected: where2comm's
        # results/null_gate/ pair is job 558438 on the SUPERSEDED OPV2V
        # pairing (ap50 0.857), and silently adopting it would reintroduce
        # the domain error the sweep was re-run to remove.
        ref, ref_src = None, None
        wj = str(wnull.get('job_id') or '') if wnull else ''
        if wnull is not None and wj:
            if clean is not None and str(clean.get('job_id') or '') == wj:
                ref, ref_src = clean, 'none/clean (same job %s)' % wj
            else:
                # null_gate/ is a sibling of the sweep root
                cand = read_bundle(os.path.join(
                    os.path.dirname(os.path.normpath(args.root)),
                    'null_gate', model, 'unwrapped'))
                if cand is not None and str(cand.get('job_id') or '') == wj:
                    ref, ref_src = cand, 'null_gate/%s/unwrapped (same job %s)' % (model, wj)
                elif cand is not None:
                    flags.append('GATE-REF-REJECTED %s: null_gate reference is '
                                 'job %s but wrapped_null is job %s -- not '
                                 'contemporaneous, refusing to use it'
                                 % (model, cand.get('job_id'), wj))
        if wnull is not None and ref is None:
            flags.append('GATE-NO-CONTEMPORANEOUS-REF %s: wrapped_null (job '
                         '%s) has no same-job unwrapped reference; the gate '
                         'CANNOT be evaluated and fails closed'
                         % (model, wj or '?'))
            manifest['wrapped_null_gate'][model] = 'MISSING (no contemporaneous reference)'
            wnull = None
            no_ref = True
        else:
            no_ref = False

        if wnull is None and not no_ref:
            flags.append('GATE-MISSING %s: no none/wrapped_null cell -- the '
                         'mandatory wrapped-null gate did not run; fault '
                         'cells are UNVERIFIED against adapter round-trip '
                         'error' % model)
            manifest['wrapped_null_gate'][model] = 'MISSING'
        elif wnull is not None:
            r50, r70 = ref['ap_50'], ref['ap_70']
            g50 = abs(wnull['ap_50'] - r50)
            g70 = abs(wnull['ap_70'] - r70)
            ok = g50 <= gate_floor and g70 <= gate_floor
            manifest['wrapped_null_gate'][model] = {
                'reference_source': ref_src,
                'unwrapped_shipped': {'ap_50': r50, 'ap_70': r70},
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
                # cosdh's driver bundles carry no n_frames field; fall back
                # to the grid's expected_frames (2170, verified at gate 1)
                # rather than emitting an empty cell figures cannot int().
                'n_frames': (n_frames_override if n_frames_override is not None
                             else bundle.get('n_frames') or exp_frames or ''),
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
                if (model, injector) in NOT_APPLICABLE:
                    # NOT a missing cell and NOT a 0.0 result -- excluded from
                    # rows entirely so it cannot enter any ordering.
                    na.append('%s/%s/%s' % (model, injector, tier))
                    continue
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
    manifest['not_applicable_cells'] = na
    if na:
        flags.append('NOT-APPLICABLE (%d cells, excluded from all rankings): %s' % (len(na), ', '.join(sorted(na))))
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
