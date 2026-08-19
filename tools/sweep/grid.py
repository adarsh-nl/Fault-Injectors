"""
grid.py
-------
The sweep grid: single source of truth for models, injectors, tiers and their
FaultSpec mappings. The cells files the sbatch jobs consume are GENERATED
from this module (``python tools/sweep/grid.py <model> <part>``), and the
aggregator imports it, so the job definitions and the results schema cannot
drift apart.

Design decisions recorded here because the numbers depend on them:

* Each model runs at its own SHIPPED wild_setting; faults inject on top.
  The clean control is the UNWRAPPED official pipeline (spec ``null``) --
  the purest "model as shipped" -- and the wrapper-null offset was measured
  within the determinism floor (job 557236), so wrapped fault cells are
  comparable to the unwrapped control.
* Scopes: pose_error / latency / agent_drop / missing_lidar are V2X-link
  faults -> non-ego only (latency and the protections are structural).
  points_reduce / lidar_fog / lidar_snow are sensor/environment faults ->
  all agents including the ego.
* lidar_snow carries ``mount_height=1.9`` (CARLA LiDAR mount on OPV2V/V2XSet;
  the injector default 1.10 is Griffin's and mis-seeds the ground fit that
  calibrates the noise-floor removal).
* One base seed (1234) for every cell; per-sample draws derive from
  ``SeedSequence(spawn_key=(frame, agent, stage))``, so cells are independent
  and every logged draw is re-derivable.
"""

import json

SEED = 1234

TIER_NAMES = ('mild', 'moderate', 'severe')

# injector -> (severity_unit, [(severity_value, spec_kwargs), x3])
INJECTORS = {
    'pose_error': ('pose_sigma', [
        ('0.2m/0.2deg', {'pose_error': {'sigma_xy': 0.2, 'sigma_heading': 0.2}}),
        ('0.4m/0.4deg', {'pose_error': {'sigma_xy': 0.4, 'sigma_heading': 0.4}}),
        ('0.6m/0.6deg', {'pose_error': {'sigma_xy': 0.6, 'sigma_heading': 0.6}}),
    ]),
    'latency': ('latency_ms', [
        ('100ms', {'latency': {'mu_delay': 0.1, 'sigma_jitter': 0.0}}),
        ('200ms', {'latency': {'mu_delay': 0.2, 'sigma_jitter': 0.0}}),
        ('300ms', {'latency': {'mu_delay': 0.3, 'sigma_jitter': 0.0}}),
    ]),
    'agent_drop': ('drop_frac', [
        ('0.25', {'agent_drop': {'p_drop': 0.25}}),
        ('0.50', {'agent_drop': {'p_drop': 0.50}}),
        ('0.75', {'agent_drop': {'p_drop': 0.75}}),
    ]),
    'missing_modality': ('drop_frac', [
        ('0.25', {'missing_lidar': {'p_drop_lidar': 0.25}}),
        ('0.50', {'missing_lidar': {'p_drop_lidar': 0.50}}),
        ('0.75', {'missing_lidar': {'p_drop_lidar': 0.75}}),
    ]),
    'points_reduce': ('keep_frac', [
        ('0.30', {'points_reduce': {'severity': 1}}),
        ('0.20', {'points_reduce': {'severity': 2}}),
        ('0.10', {'points_reduce': {'severity': 3}}),
    ]),
    'lidar_fog': ('mc_severity', [
        ('sev1', {'lidar_fog': {'severity': 1}}),
        ('sev2', {'lidar_fog': {'severity': 2}}),
        ('sev3', {'lidar_fog': {'severity': 3}}),
    ]),
    'lidar_snow': ('mc_severity', [
        ('sev1', {'lidar_snow': {'severity': 1, 'mount_height': 1.9}}),
        ('sev2', {'lidar_snow': {'severity': 2, 'mount_height': 1.9}}),
        ('sev3', {'lidar_snow': {'severity': 3, 'mount_height': 1.9}}),
    ]),
}

MODELS = {
    'cobevt': {
        'dataset': 'v2xset', 'setting': 'perfect',
        'data_root': '/datasets/eemcs/ps/cv/opencood/v2xset/test',
        'model_dir': '~/opencood-eval/cobevt',
        'expected_frames': 2834, 'graded': True,
        'published': {'ap50': 0.849, 'ap70': 0.660},
    },
    'where2comm': {
        # CORRECTED 2026-08-08. The checkpoint is V2XSET-trained: its md5
        # (4beff417ffe6c62d76c88acaff63d32c) is identical to the one inside
        # /datasets/.../v2xset_checkpoints/point_pillar_where2comm_v2xset.zip,
        # and its config is byte-identical to that archive's. It was
        # previously paired with OPV2V test -- a domain mismatch. The
        # config's `root_dir: /data/opv2x/train` is the author's shorthand,
        # NOT evidence of OPV2V.
        'dataset': 'v2xset', 'setting': 'noisy',
        'data_root': '/datasets/eemcs/ps/cv/opencood/v2xset/test',
        'model_dir': '~/opencood-eval/where2comm',
        'expected_frames': 2834,
        # Ungraded for PROVENANCE, not for a missing table entry: V2XSet is
        # not a Where2comm dataset (the paper uses OPV2V, V2X-Sim, DAIR-V2X,
        # CoPerception-UAVs), so this is a THIRD-PARTY retraining published
        # as a V2XSet baseline, not an author release. No published number
        # exists that is the Where2comm authors' own.
        'graded': False,
        'published': None,
    },
    'cosdh': {
        # CoSDH (CVPR 2025) -- an OpenCOOD FORK, own env/driver/adapter.
        # Ships noise_setting {add_noise: false}: PERFECT, groups with cobevt.
        # NOT gradeable against the paper's 0.9683/0.9299 -- the released
        # checkpoints are retrained (release README). Measured clean baseline
        # (gate 1, job 561691): 0.9660 / 0.9289.
        'dataset': 'opv2v', 'setting': 'perfect',
        'data_root': '/datasets/eemcs/ps/cv/opencood/opv2v/test',
        'model_dir': '/datasets/eemcs/ps/cv/opencood/cosdh_checkpoints/opv2v_cosdh',
        'expected_frames': 2170,
        'graded': False,
        'published': None,
    },
    'v2xvit': {
        'dataset': 'v2xset', 'setting': 'noisy',
        'data_root': '/datasets/eemcs/ps/cv/opencood/v2xset/test',
        'model_dir': '/datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/v2x-vit',
        'expected_frames': 2834, 'graded': True,
        'published': {'ap50': 0.836, 'ap70': 0.614},
    },
}

# job partitions: which injectors run in which job (wall-time balancing;
# snow is ~3 h per cell, everything else ~25 min)
PARTS = {
    'a': ['pose_error', 'latency', 'agent_drop', 'missing_modality',
          'points_reduce'],
    'b': ['lidar_fog'],
    'c': ['lidar_snow'],
}


def cells(model, part):
    """Yield (outdir, spec_json) lines for one job. Every job starts with a
    clean pass (part a additionally runs the determinism repeat AND the
    mandatory wrapped-null gate).

    WRAPPED-NULL GATE (mandatory, non-skippable -- added 2026-08-06 after the
    pose-contamination bug). ``none/clean`` is the UNWRAPPED official
    pipeline at the model's SHIPPED wild_setting; ``none/wrapped_null`` is the
    wrapper carrying an empty pipeline at that same shipped setting. The two
    must agree within the model's determinism floor.

    Why it must compare against the SHIPPED baseline and not a forced-clean
    run: the bug it exists to catch was the adapter silently discarding
    OpenCOOD's own ``add_loc_noise`` perturbation (applied to a local copy and
    never written into ``params['lidar_pose']``). A forced-clean reference has
    no such noise, so it would agree with the broken adapter and the golden
    would encode the very defect it guards. Measured before the fix:
    0.177 (Where2comm) / 0.205 (V2X-ViT) matrix error with NO fault injected;
    CoBEVT was 0.000 only because it ships Perfect.
    """
    out = [('none/clean_%s' % part if part != 'a' else 'none/clean', 'null')]
    if part == 'a':
        out.append(('none/clean_rep', 'null'))
        out.append(('none/wrapped_null', '{}'))    # mandatory gate
    for inj in PARTS[part]:
        unit, tiers = INJECTORS[inj]
        for tier_name, (value, kwargs) in zip(TIER_NAMES, tiers):
            out.append(('%s/%s' % (inj, tier_name),
                        json.dumps(kwargs, sort_keys=True)))
    return out


if __name__ == '__main__':
    import sys
    model, part = sys.argv[1], sys.argv[2]
    for outdir, spec in cells(model, part):
        print('%s|%s' % (outdir, spec))
