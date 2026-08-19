"""Collect (Z, Y) for information-quality analysis from an OpenCOOD model.

Answers the mechanism question behind the benchmark's headline finding
(losing collaborators hurts far less than receiving corrupted data):

    I(F_fused; Y) - I(F_ego; Y)

Under agent_drop the gain should shrink toward zero -- fewer collaborators
means less ADDED information, but nothing corrupted. Under pose error it
may go NEGATIVE -- fusion discarding good ego information while folding in
misinformation it cannot distinguish from signal.

TWO PROTOCOL DECISIONS, both load-bearing, both user decisions:

1. **Y IS HELD FIXED AT ITS CLEAN VALUE across every condition.** Computed
   once on the clean run and reused verbatim. If Y derived from the union
   GT under each condition, agent_drop would change BOTH the
   representation AND the target -- the GT-union confound already
   documented in the sweep manifest -- and would confound exactly the
   comparison this experiment exists to make. Same target, different
   representation, so any MI change is attributable to the fault.
2. **Y is a coarse 8x8 spatial histogram of `pos_equal_one`** (64-dim, one
   row per frame). A scene-summary Y throws away the BEV alignment we get
   for free; a per-cell Y forces per-cell pooling of Z and changes N from
   frame count to cell count.

EGO POOLING: the pre-fusion tensor is (n_agents, C, H, W) -- dim 0 is the
AGENT axis, so `global_pool`'s mean over dims (0,2,3) would average ego
and collaborators together and destroy the very ego-vs-fused distinction
the experiment measures. `ego_pool` selects row 0 and spatially averages.
OpenCOOD guarantees ego is row 0 (intermediate_fusion_dataset.py asserts
"The first element in the OrderedDict must be ego"); we ASSERT it at
runtime rather than trusting it.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/nanjaiyalathaa/Fault-Injectors")

from src.adapters.runtime import FaultSpec, make_faulty_dataset   # noqa: E402
from src.info_quality.feature_extraction import (                 # noqa: E402
    FeatureCollector, FeatureSet)

P = lambda *a: print(*a, flush=True)                              # noqa: E731

# severe tier of each injector, matching tools/sweep/grid.py exactly
CONDITIONS = {
    'clean':      None,
    'pose_error': {'pose_error': {'sigma_xy': 0.6, 'sigma_heading': 0.6}},
    'agent_drop': {'agent_drop': {'p_drop': 0.75}},
}


# ── pooling ─────────────────────────────────────────────────────────────
_EGO_SEEN = {'n': 0, 'agents': []}


def ego_pool(t: torch.Tensor) -> torch.Tensor:
    """(n_agents, C, H, W) -> (C,) taking the EGO row only.

    NOT a mean over dim 0: that is the agent axis here, and averaging it
    would mix ego with collaborators, which is precisely the distinction
    under test.
    """
    if t.dim() != 4:
        raise RuntimeError('ego_pool expected a 4-D (agents, C, H, W) '
                           'tensor, got %dD %s' % (t.dim(), tuple(t.shape)))
    _EGO_SEEN['n'] += 1
    _EGO_SEEN['agents'].append(int(t.shape[0]))
    return t[0].mean(dim=(1, 2)).detach().to('cpu', torch.float32)


def fused_pool(t: torch.Tensor) -> torch.Tensor:
    """(B=1, C, H, W) -> (C,). Batch is 1, so the dim-0 mean is a no-op and
    the spatial mean is the intended reduction."""
    if t.dim() != 4:
        raise RuntimeError('fused_pool expected 4-D, got %dD %s'
                           % (t.dim(), tuple(t.shape)))
    if t.shape[0] != 1:
        raise RuntimeError('fused_pool expects batch 1, got %d' % t.shape[0])
    return t[0].mean(dim=(1, 2)).detach().to('cpu', torch.float32)


# ── target ──────────────────────────────────────────────────────────────
def make_label_fn(grid, store):
    """8x8 occupancy histogram of pos_equal_one, 64-dim, one row per frame.

    NO try/except: a wrong key path must fail on the first sample, not
    produce a constant Y that reads as a plausible null across every
    condition. FeatureCollector additionally asserts Y is non-constant.
    """
    gh, gw = grid

    def label_fn(batch):
        peo = batch['ego']['label_dict']['pos_equal_one']
        t = peo if torch.is_tensor(peo) else torch.as_tensor(peo)
        t = t.detach().float().cpu()
        # (B, H, W, A) from collate; drop batch, sum over the anchor axis
        if t.dim() == 4:
            t = t[0]
        occ = t.sum(dim=-1)                       # (H, W) anchor-positive count
        H, W = occ.shape
        ph, pw = H // gh, W // gw
        # crop to a multiple of the pool factor, then block-sum
        occ = occ[:ph * gh, :pw * gw]
        hist = occ.reshape(gh, ph, gw, pw).sum(dim=(1, 3))   # (gh, gw)
        store['H'], store['W'] = H, W
        return hist.reshape(-1).numpy().astype(np.float32)

    return label_fn


# ── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--condition', required=True, choices=list(CONDITIONS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--n_samples', type=int, default=None)
    ap.add_argument('--grid', type=int, default=8)
    ap.add_argument('--ns', default='opencood')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ocds = importlib.import_module(args.ns + '.data_utils.datasets')
    yaml_utils = importlib.import_module(args.ns + '.hypes_yaml.yaml_utils')
    train_utils = importlib.import_module(args.ns + '.tools.train_utils')

    base_name = 'IntermediateFusionDataset'
    base_cls = ocds.__all__[base_name]
    cond = CONDITIONS[args.condition]
    log_dir = os.path.join(os.path.dirname(args.out), 'injection_%s'
                           % args.condition)
    if cond is not None:
        os.makedirs(log_dir, exist_ok=True)
        spec = FaultSpec(seed=args.seed, log_dir=log_dir, **cond)
        ocds.__all__[base_name] = make_faulty_dataset(base_cls, spec)
        P('[mi] condition=%s wrapped -> %s' % (args.condition,
                                               ocds.__all__[base_name].__name__))
    else:
        spec = None
        P('[mi] condition=clean, UNWRAPPED official dataset')

    hypes = yaml_utils.load_yaml(os.path.join(args.model_dir, 'config.yaml'))
    P('[mi] wild_setting (shipped, unmodified): %s'
      % hypes.get('wild_setting'))

    opencood_dataset = ocds.build_dataset(hypes, visualize=False, train=False)

    # ── RUNTIME EGO-ORDER ASSERTION ─────────────────────────────────────
    # ego_pool takes row 0 of the per-agent tensor. That is only the EGO
    # because OpenCOOD orders base_data_dict ego-first
    # (intermediate_fusion_dataset.py: "The first element in the OrderedDict
    # must be ego"). We do not TRUST that invariant, we CHECK it on every
    # frame -- and it matters more here than usual, because the fault
    # wrapper mutates base_data_dict (agent_drop deletes entries) between
    # OpenCOOD's assertion and the tensor we pool.
    _orig_retrieve = opencood_dataset.retrieve_base_data
    ego_checks = {'n': 0}

    def _retrieve_checked(idx, *a, **kw):
        base = _orig_retrieve(idx, *a, **kw)
        keys = list(base)
        if not keys:
            raise RuntimeError('frame %s: base_data_dict is EMPTY' % idx)
        if not base[keys[0]].get('ego'):
            raise RuntimeError(
                'frame %s: first entry %r is NOT the ego (ego=%r). ego_pool '
                'takes row 0 of the per-agent tensor and would silently pool '
                'a COLLABORATOR as if it were the ego.'
                % (idx, keys[0], [k for k in keys if base[k].get('ego')]))
        if sum(1 for k in keys if base[k].get('ego')) != 1:
            raise RuntimeError('frame %s: expected exactly one ego, got %d'
                               % (idx, sum(1 for k in keys
                                           if base[k].get('ego'))))
        ego_checks['n'] += 1
        return base

    opencood_dataset.retrieve_base_data = _retrieve_checked
    loader = torch.utils.data.DataLoader(
        opencood_dataset, batch_size=1, num_workers=4, shuffle=False,
        collate_fn=opencood_dataset.collate_batch_test, pin_memory=False,
        drop_last=False)
    P('[mi] dataset frames: %d' % len(opencood_dataset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.to(device).eval()
    P('[mi] model %s on %s' % (type(model).__name__, device))

    # ── taps. shrink_conv / naive_compressor are PER-AGENT (n, C, H, W);
    # fusion_net is POST-fusion (1, C, H, W). Different pools, deliberately.
    taps, pools = {}, {}
    if hasattr(model, 'shrink_conv'):
        taps['Z_ego_shrink'] = model.shrink_conv
        pools['Z_ego_shrink'] = ego_pool
    if hasattr(model, 'naive_compressor'):
        taps['Z_ego_compressed'] = model.naive_compressor
        pools['Z_ego_compressed'] = ego_pool
    taps['Z_fused'] = model.fusion_net
    pools['Z_fused'] = fused_pool
    P('[mi] taps: %s' % list(taps))

    store = {}
    label_fn = make_label_fn((args.grid, args.grid), store)

    def forward_fn(m, batch):
        b = train_utils.to_device(batch, device)
        return m(b['ego'])

    collector = FeatureCollector(model, taps=taps, taps_pool=pools)
    fs = collector.collect(loader, forward_fn=forward_fn, label_fn=label_fn,
                           n_samples=args.n_samples, progress=False)
    collector.remove()

    P('[mi] anchor grid seen: %dx%d -> Y pooled to %dx%d = %d dims'
      % (store.get('H', -1), store.get('W', -1), args.grid, args.grid,
         args.grid ** 2))
    ag = np.asarray(_EGO_SEEN['agents'])
    P('[mi] ego_pool fired %d times; agents/frame min=%d mean=%.2f max=%d'
      % (_EGO_SEEN['n'], ag.min(), ag.mean(), ag.max()))
    P('[mi] ego-order assertion passed on %d frames (workers may re-check '
      'in their own processes; a failure raises there)' % ego_checks['n'])

    fs.save(args.out)
    P('[mi] saved %s' % args.out)
    for k, v in fs.features.items():
        P('   %-20s %s' % (k, v.shape))
    P('   %-20s %s' % ('Y', fs.Y.shape))

    sd = fs.Y.std(axis=0)
    meta = {
        'condition': args.condition,
        'spec': spec.as_dict() if spec else None,
        'seed': args.seed,
        'n_frames': int(fs.Y.shape[0]),
        'anchor_grid': [store.get('H'), store.get('W')],
        'y_dims': int(fs.Y.shape[1]),
        'y_nondegenerate_cols': int((sd > 0).sum()),
        'y_col_var_min': float(sd.min() ** 2),
        'y_col_var_max': float(sd.max() ** 2),
        'y_col_var_mean': float((sd ** 2).mean()),
        'y_frac_frames_zero_occupancy': float((fs.Y.sum(axis=1) == 0).mean()),
        'y_total_occupancy_mean': float(fs.Y.sum(axis=1).mean()),
        'agents_per_frame_mean': float(ag.mean()),
        'agents_per_frame_min': int(ag.min()),
        'taps': list(fs.features),
    }
    with open(args.out.replace('.npz', '_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    P('[mi] Y stats: %d/%d non-degenerate cols | var min/mean/max '
      '%.4g/%.4g/%.4g | frames with ZERO occupancy %.3f | mean total occ %.2f'
      % (meta['y_nondegenerate_cols'], meta['y_dims'], meta['y_col_var_min'],
         meta['y_col_var_mean'], meta['y_col_var_max'],
         meta['y_frac_frames_zero_occupancy'], meta['y_total_occupancy_mean']))
    P('MI COLLECT DONE')


if __name__ == '__main__':
    main()
