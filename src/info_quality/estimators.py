"""
estimators.py
-------------
Architecture- and dataset-agnostic mutual information estimators.

Information quality, in this project, is operationalised as the mutual
information between a learned representation Z and a task-relevant target Y:

    I(Z; Y)  =  how much task information the representation carries.

Under fault injection we estimate I(Z; Y) for the same representation under
clean and corrupted inputs and read off how much information survives. A single
fusion gain is summarised as

    delta_I  =  I(Z_fused; Y)  -  max( I(Z_a; Y), I(Z_b; Y) )

i.e. the information the fused representation holds beyond the best single
modality. delta_I > 0 means fusion is synergistic; delta_I <= 0 means it is
redundant or lossy.

Why two estimators?
-------------------
I(Z; Y) has no closed form for continuous neural features; it must be estimated,
and every estimator is biased. Reporting two estimators with different biases
turns "fusion adds information" into "fusion adds information under both
InfoNCE and SMILE", which is far harder for a reviewer to dismiss as an
estimator artefact.

  InfoNCE (contrastive lower bound, van den Oord et al. 2018)
      I >= log(K) - L_NCE, where L_NCE is the contrastive cross-entropy over a
      K-way classification of matched (z_i, y_i) against in-batch negatives.
      Stable and low variance, but the bound is capped at log(K): with K=N=81
      it cannot report more than log(81) ~= 4.39 nats no matter the true MI.

  SMILE (clipped variational lower bound, Song & Ermon 2020)
      A variance-reduced repair of MINE. MINE's partition term uses a detached
      EMA denominator, so the network can drive T -> infinity with no gradient
      penalty (a "death spiral"). SMILE clips the log density ratio to
      [-clip, +clip] inside a fully differentiable denominator, bounding the
      gradient and removing the divergence. It has no log(K) ceiling.

Bias and the comparison protocol
--------------------------------
Both bounds are evaluated in-sample (trained and read off on the same N), which
is optimistic. This is acceptable here because the quantity of interest is
*relative* (delta_I, clean vs faulty), and the protocol is held fixed across
conditions, so the systematic bias cancels in the difference. Absolute MI
values should be read as lower bounds, not point estimates.

Agnosticism
-----------
Everything in this module consumes plain (N, d) NumPy arrays. It has no
knowledge of any model or dataset. The same code estimates I(Z; Y) for a
BEVFusion BEV feature on nuScenes, a Griffin drone-side voxel feature, or a
synthetic Gaussian (see `correlated_gaussians`, used to validate the
estimators against a closed-form ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import prepare, set_seed


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class MIResult:
    """
    Outcome of one mutual information estimation.

    Attributes
    ----------
    mi_nats   : float  estimated I(Z; Y) in nats (lower bound).
    estimator : str    'infonce' or 'smile'.
    n_samples : int    N used.
    history   : list of float  per-step MI estimates after warmup (SMILE) or
                empty (InfoNCE reports a single final bound).
    meta      : dict   estimator-specific bookkeeping (epochs, temperature, ...).
    """
    mi_nats: float
    estimator: str
    n_samples: int
    history: List[float] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _split(n: int, holdout: float, seed: Optional[int]):
    """
    Seeded train/eval index split. Returns (train_idx, eval_idx) as int arrays.

    holdout is the fraction of samples held out for evaluating the bound. With
    holdout=0 the same indices are returned for both (in-sample evaluation).
    """
    if holdout <= 0.0:
        idx = np.arange(n)
        return idx, idx
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_ev = max(2, int(round(holdout * n)))
    n_ev = min(n_ev, n - 2)  # leave at least 2 for training
    return perm[n_ev:], perm[:n_ev]


# ── InfoNCE ─────────────────────────────────────────────────────────────────

class _InfoNCECritic(nn.Module):
    """
    Separable critic with projection heads and a temperature.

    Scores a matched pair by cosine similarity of independently projected
    z and y, scaled by 1/tau. This is the standard CPC / SimCLR critic.
    """

    def __init__(self, z_dim: int, y_dim: int, proj_dim: int = 64,
                 temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
        self.proj_z = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(), nn.Linear(128, proj_dim))
        self.proj_y = nn.Sequential(
            nn.Linear(y_dim, 64), nn.ReLU(), nn.Linear(64, proj_dim))

    def logits(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        zp = F.normalize(self.proj_z(z), dim=-1)
        yp = F.normalize(self.proj_y(y), dim=-1)
        return torch.matmul(zp, yp.T) / self.tau

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.logits(z, y)
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def mi_bound(self, z: torch.Tensor, y: torch.Tensor) -> float:
        # I >= log(K) - L_NCE, evaluated over the full set (K = N).
        loss = self.forward(z, y).item()
        return float(np.log(z.size(0)) - loss)


class InfoNCEEstimator:
    """
    InfoNCE contrastive lower bound on I(Z; Y).

    Parameters
    ----------
    proj_dim    : int   projection head width.
    temperature : float lower tau -> harder negatives -> tighter (higher) bound.
    epochs      : int   training passes over the data.
    batch_size  : int   contrastive batch; the in-batch negatives set the bound
                        ceiling during training (final read-off uses full N).
    lr          : float Adam learning rate.
    holdout     : float fraction of samples held out to evaluate the bound on
                        unseen pairs. 0 = in-sample (optimistic; fine for the
                        small-N regime and for relative comparisons under a
                        fixed protocol). > 0 strongly recommended when N is
                        large enough to spare a split, since in-sample InfoNCE
                        can report a sizeable positive bound even for
                        independent data (the critic memorises the pairing).
    seed        : int   reproducible run, or None for a fresh stream.
    device      : 'cuda' / 'cpu' / None (auto-detect).
    """

    def __init__(self, proj_dim: int = 64, temperature: float = 0.07,
                 epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                 holdout: float = 0.0, seed: Optional[int] = 0,
                 device: Optional[str] = None):
        self.proj_dim = proj_dim
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.holdout = holdout
        self.seed = seed
        self.device = device

    def estimate(self, Z: np.ndarray, Y: np.ndarray,
                 pca_dims: Optional[int] = None) -> MIResult:
        """
        Estimate I(Z; Y) in nats. Z, Y are (N, d) arrays with aligned rows.
        """
        set_seed(self.seed)
        Z, Y = prepare(Z, Y, pca_dims)
        device = _device(self.device)

        tr, ev = _split(Z.shape[0], self.holdout, self.seed)
        # Index with numpy, then convert with an explicit dtype. This avoids
        # asking torch to infer the dtype of a numpy int64 index array, which
        # fails on builds where the torch<->numpy bridge is version-mismatched.
        Ztr = torch.as_tensor(Z[tr], dtype=torch.float32, device=device)
        Ytr = torch.as_tensor(Y[tr], dtype=torch.float32, device=device)
        Zev = torch.as_tensor(Z[ev], dtype=torch.float32, device=device)
        Yev = torch.as_tensor(Y[ev], dtype=torch.float32, device=device)
        N_tr = Ztr.size(0)
        bs = min(self.batch_size, N_tr)

        critic = _InfoNCECritic(Z.shape[1], Y.shape[1],
                                self.proj_dim, self.temperature).to(device)
        opt = torch.optim.Adam(critic.parameters(), lr=self.lr)

        critic.train()
        for _ in range(self.epochs):
            perm = torch.randperm(N_tr, device=device)
            for start in range(0, N_tr - bs + 1, bs):  # drop last partial batch
                idx = perm[start:start + bs]
                opt.zero_grad()
                critic(Ztr[idx], Ytr[idx]).backward()
                opt.step()

        critic.eval()
        mi = critic.mi_bound(Zev, Yev)
        return MIResult(
            mi_nats=mi, estimator='infonce', n_samples=Z.shape[0],
            meta=dict(epochs=self.epochs, temperature=self.temperature,
                      batch_size=bs, pca_dims=pca_dims, holdout=self.holdout,
                      n_eval=int(Zev.size(0)), ceiling_nats=float(np.log(Zev.size(0)))),
        )


# ── SMILE ───────────────────────────────────────────────────────────────────

class _SMILENet(nn.Module):
    """Statistics network T(z, y) = MLP([z; y]) producing a scalar score."""

    def __init__(self, z_dim: int, y_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, y], dim=-1)).squeeze(-1)


def _smile_lower_bound(t_joint: torch.Tensor, t_marg: torch.Tensor,
                       clip: float) -> torch.Tensor:
    """
    SMILE lower bound from joint and marginal scores.

    I_SMILE = E_joint[T] - log E_marg[ exp( clamp(T, -clip, +clip) ) ]

    Clamping the log density ratio T to [-clip, +clip] before exponentiating
    bounds the partition gradient, which is exactly what stops MINE's divergence.

    Parameters
    ----------
    t_joint : (B,)    scores on matched pairs (z_i, y_i).
    t_marg  : (B, M)  scores on mismatched pairs (z_i, y_j).
    clip    : float   clip threshold in nats.
    """
    log_denom = t_marg.clamp(-clip, clip).exp().mean(dim=1).log()  # (B,)
    return (t_joint - log_denom).mean()


class SMILEEstimator:
    """
    SMILE clipped variational lower bound on I(Z; Y).

    Uses many samples as negatives each step (M marginal scores per anchor),
    which sharply reduces variance versus in-batch-only negatives. M is capped
    at `max_negatives` so cost stays O(B * M) rather than O(B * N): for the
    small-N regime this module targets (tens to a few hundred features) every
    sample is used; for large N the negatives are subsampled, keeping memory
    bounded.

    Parameters
    ----------
    hidden        : int   statistics-network width.
    clip          : float SMILE clip threshold (nats); smaller -> lower
                          variance, looser bound. 5.0 is a standard default.
    epochs        : int   optimisation steps.
    batch_size    : int   number of anchor samples per step.
    lr            : float Adam learning rate.
    warmup        : int   steps to skip before recording MI estimates.
    avg_last      : int   number of trailing estimates to average into report.
    eval_every    : int   record a full-data MI estimate every k steps after
                          warmup (k > 1 keeps long runs cheap).
    max_negatives : int   cap on negatives per anchor (bounds O(N^2) memory).
    grad_clip     : float max gradient norm.
    holdout       : float fraction held out to evaluate the bound on unseen
                          pairs (see InfoNCEEstimator). 0 = in-sample.
    floor_zero    : bool  clamp the reported estimate at 0 (MI is non-negative).
                          Off by default so the raw lower bound (which can dip
                          slightly negative from variance) is reported honestly.
    seed          : int   reproducible run, or None.
    device        : 'cuda' / 'cpu' / None (auto-detect).
    """

    def __init__(self, hidden: int = 256, clip: float = 5.0,
                 epochs: int = 500, batch_size: int = 64, lr: float = 2e-4,
                 warmup: int = 100, avg_last: int = 50, eval_every: int = 5,
                 max_negatives: int = 512, grad_clip: float = 10.0,
                 holdout: float = 0.0, floor_zero: bool = False,
                 seed: Optional[int] = 0, device: Optional[str] = None):
        self.hidden = hidden
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup = warmup
        self.avg_last = avg_last
        self.eval_every = max(1, eval_every)
        self.max_negatives = max_negatives
        self.grad_clip = grad_clip
        self.holdout = holdout
        self.floor_zero = floor_zero
        self.seed = seed
        self.device = device

    @torch.no_grad()
    def _full_estimate(self, net: _SMILENet, Z_anchor: torch.Tensor,
                       Y_anchor: torch.Tensor, Y_neg: torch.Tensor,
                       anchor_chunk: int = 256) -> float:
        """
        MI estimate on a set of anchors against a fixed negative target set,
        chunked over anchors so peak memory is O(anchor_chunk * M).
        """
        N = Z_anchor.size(0)
        M = Y_neg.size(0)
        t_joint = net(Z_anchor, Y_anchor).mean()              # scalar
        logdenom_sum = Z_anchor.new_zeros(())
        for start in range(0, N, anchor_chunk):
            zc = Z_anchor[start:start + anchor_chunk]          # (c, d_z)
            c = zc.size(0)
            z_rep = zc.unsqueeze(1).expand(-1, M, -1).reshape(c * M, -1)
            y_rep = Y_neg.unsqueeze(0).expand(c, -1, -1).reshape(c * M, -1)
            t_marg = net(z_rep, y_rep).reshape(c, M)
            logdenom_sum = logdenom_sum + t_marg.clamp(-self.clip, self.clip).exp().mean(dim=1).log().sum()
        mi = (t_joint - logdenom_sum / N).item()
        return max(0.0, mi) if self.floor_zero else mi

    def estimate(self, Z: np.ndarray, Y: np.ndarray,
                 pca_dims: Optional[int] = None) -> MIResult:
        """
        Estimate I(Z; Y) in nats. Z, Y are (N, d) arrays with aligned rows.
        """
        set_seed(self.seed)
        Z, Y = prepare(Z, Y, pca_dims)
        device = _device(self.device)

        tr, ev = _split(Z.shape[0], self.holdout, self.seed)
        # Index with numpy, then convert with an explicit dtype. This avoids
        # asking torch to infer the dtype of a numpy int64 index array, which
        # fails on builds where the torch<->numpy bridge is version-mismatched.
        Ztr = torch.as_tensor(Z[tr], dtype=torch.float32, device=device)
        Ytr = torch.as_tensor(Y[tr], dtype=torch.float32, device=device)
        Zev = torch.as_tensor(Z[ev], dtype=torch.float32, device=device)
        Yev = torch.as_tensor(Y[ev], dtype=torch.float32, device=device)
        N_tr = Ztr.size(0)
        bs = min(self.batch_size, N_tr)
        M = min(self.max_negatives, N_tr)
        M_ev = min(self.max_negatives, Zev.size(0))
        # Fixed negative target set for the eval estimate (stable across steps).
        eval_neg = Yev[torch.randperm(Zev.size(0), device=device)[:M_ev]]

        net = _SMILENet(Z.shape[1], Y.shape[1], self.hidden).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        history: List[float] = []
        net.train()
        for step in range(self.epochs):
            idx = torch.randperm(N_tr, device=device)[:bs]
            zb, yb = Ztr[idx], Ytr[idx]                        # anchors
            B = zb.size(0)
            y_neg = Ytr[torch.randperm(N_tr, device=device)[:M]]  # fresh negatives

            t_joint = net(zb, yb)                              # (B,)
            zb_exp = zb.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
            y_exp = y_neg.unsqueeze(0).expand(B, -1, -1).reshape(B * M, -1)
            t_marg = net(zb_exp, y_exp).reshape(B, M)          # (B, M)

            loss = -_smile_lower_bound(t_joint, t_marg, self.clip)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            sched.step()

            if step >= self.warmup and (step - self.warmup) % self.eval_every == 0:
                net.eval()
                history.append(self._full_estimate(net, Zev, Yev, eval_neg))
                net.train()

        if history:
            tail = history[-self.avg_last:] if len(history) >= self.avg_last else history
            mi = float(np.mean(tail))
        else:
            net.eval()
            mi = self._full_estimate(net, Zev, Yev, eval_neg)

        return MIResult(
            mi_nats=mi, estimator='smile', n_samples=Z.shape[0], history=history,
            meta=dict(epochs=self.epochs, clip=self.clip, batch_size=bs,
                      warmup=self.warmup, avg_last=self.avg_last,
                      pca_dims=pca_dims, holdout=self.holdout,
                      n_eval=int(Zev.size(0)), floor_zero=self.floor_zero),
        )


# ── Convenience: fusion gain over a set of representations ───────────────────

def delta_information(mi_by_name: dict, fused_key: str,
                      unimodal_keys: List[str]) -> float:
    """
    Fusion information gain: I(fused; Y) - max_i I(unimodal_i; Y).

    Parameters
    ----------
    mi_by_name    : dict name -> MI in nats (e.g. {'camera':.., 'lidar':.., 'fused':..}).
    fused_key     : key of the fused representation.
    unimodal_keys : keys of the single-modality representations to compare against.

    Returns
    -------
    float delta_I in nats. Positive => fusion is synergistic.
    """
    best_uni = max(mi_by_name[k] for k in unimodal_keys)
    return mi_by_name[fused_key] - best_uni


# ── Validation utility: data with a closed-form ground-truth MI ──────────────

def correlated_gaussians(n: int, dim: int, rho: float,
                         seed: Optional[int] = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Sample (Z, Y) jointly Gaussian with per-dimension correlation rho.

    For independent standard-normal dimensions coupled at correlation rho, the
    mutual information has the closed form

        I(Z; Y) = -(dim / 2) * log(1 - rho^2)   nats,

    which gives the estimators a ground truth to be checked against. This is the
    standard benchmark used in the InfoNCE / MINE / SMILE literature.

    Parameters
    ----------
    n    : int   number of samples.
    dim  : int   dimensionality of each of Z and Y.
    rho  : float per-dimension correlation in (-1, 1).
    seed : int   reproducible draw, or None.

    Returns
    -------
    (Z, Y, true_mi_nats).
    """
    if not (-1.0 < rho < 1.0):
        raise ValueError('rho must lie strictly in (-1, 1).')
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, dim)).astype(np.float32)
    noise = rng.standard_normal((n, dim)).astype(np.float32)
    Y = rho * Z + np.sqrt(1.0 - rho ** 2) * noise
    true_mi = -0.5 * dim * np.log(1.0 - rho ** 2)
    return Z, Y, float(true_mi)
