"""Human-readable reg_dim verification report -- numbers, not "3 passed".

Run from the repo root by corabench/slurm/verify_regdim_branch.sbatch. Kept as
a real FILE rather than a heredoc inside the sbatch: job 551932 hung for its
full wall-clock at a `python - <<'PY'` block nested in a pipeline, and the
sbatch header now forbids that pattern.

Prints the encode -> decode round trip at both widths so the yaw recovery can
be eyeballed directly, then the assertion behaviour for the three cases. Exits
non-zero if any check fails, so the sbatch can gate on it.
"""

from __future__ import annotations

import numpy as np
import torch

from cpbench.data.postprocessing import BoxDecoder
from cpbench.data.preprocessing import AnchorGenerator, GridSpec, TargetAssigner
from corabench.scripts.common import assert_reg_dim_consistent
from corabench.training.losses import CoRALoss

_FAILURES: list = []


def _grid() -> GridSpec:
    return GridSpec(voxel_size=(0.4, 0.4),
                    point_range=(-70.4, -40.0, -3.0, 70.4, 40.0, 1.0))


def _encode_decode(reg_dim: int, delta_deg: float) -> float:
    """Encode a GT box at anchor_yaw + delta, decode it, return the recovered
    delta in degrees wrapped to (-180, 180]. Goes through the REAL BoxDecoder:
    construction alone would pass even with the cos channel never written."""
    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    gt = an.copy()
    gt[6] = an[6] + np.deg2rad(delta_deg)

    enc = TargetAssigner(ag, reg_dim=reg_dim)(gt[None, :])
    reg_t = enc["reg_target"].numpy()
    cls_t = enc["cls_target"].numpy()

    reg_map = torch.from_numpy(
        np.transpose(reg_t, (2, 3, 0, 1)).reshape(a * reg_dim, h, w))
    cls_map = torch.from_numpy(
        np.where(cls_t == 1, 1.0, 0.0).astype(np.float32).transpose(2, 0, 1))

    boxes, _ = BoxDecoder(ag, score_threshold=0.5, scores_are_logits=False,
                          reg_dim=reg_dim)(cls_map, reg_map)
    recovered = np.rad2deg(boxes[0, 6] - an[6])
    return float((recovered + 180.0) % 360.0 - 180.0)


def _check(label: str, ok: bool) -> None:
    print(f"    [{'OK  ' if ok else 'FAIL'}] {label}", flush=True)
    if not ok:
        _FAILURES.append(label)


def _round_trip_table(reg_dim: int, deltas, expect_exact: bool) -> None:
    branch = "atan2(sin, cos) + anchor" if reg_dim >= 8 \
        else "asin(clip(sin)) + anchor"
    print(f"  decode branch taken: {branch}", flush=True)
    print(f"  {'in (deg)':>10} {'out (deg)':>11} {'err':>9}", flush=True)
    for d in deltas:
        got = _encode_decode(reg_dim, d)
        err = abs(((got - d) + 180.0) % 360.0 - 180.0)
        print(f"  {d:>10.1f} {got:>11.4f} {err:>9.2e}", flush=True)
        if expect_exact:
            _check(f"reg_dim={reg_dim} delta={d} round-trips", err < 1e-3)


def case_a() -> None:
    print("\n=== CASE A  reg_dim=7 -- the four legacy packages ===", flush=True)
    _round_trip_table(7, [0.0, 20.0, -45.0, 89.0], expect_exact=True)

    # The legacy limitation, printed rather than hidden: asin cannot separate
    # 20 from 160 degrees. Pinned here so a future reader does not mistake it
    # for a regression introduced by the sin/cos change.
    got = _encode_decode(7, 160.0)
    print(f"  known 180-deg ambiguity: 160.0 decodes to {got:.4f}", flush=True)
    _check("reg_dim=7 is still 180-deg ambiguous (unchanged behaviour)",
           abs(got - 20.0) < 1e-3)

    # Channel 7 does not exist at width 7; reading it would IndexError here.
    ag = AnchorGenerator(_grid())
    h, w, a, _ = ag().shape
    BoxDecoder(ag, score_threshold=0.5, scores_are_logits=False, reg_dim=7)(
        torch.ones(a, h, w), torch.zeros(a * 7, h, w))
    _check("decode never reads channel 7 at reg_dim=7", True)

    model = _FakeModel(7, 7, 7, 7)
    got_dim = assert_reg_dim_consistent(model, _FakeDataset(7), CoRALoss(reg_dim=7))
    _check(f"assertion PASSES with everything at 7 (returned {got_dim})",
           got_dim == 7)


def case_b() -> None:
    print("\n=== CASE B  reg_dim=8 -- corabench ===", flush=True)
    # Angles outside asin's [-90, 90] range are the ones the old path could
    # not represent at all.
    _round_trip_table(8, [0.0, 20.0, 90.0, 160.0, -45.0, -135.0],
                      expect_exact=True)

    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, _a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    vecs = []
    for deg in (20.0, 160.0):
        gt = an.copy()
        gt[6] = an[6] + np.deg2rad(deg)
        enc = TargetAssigner(ag, reg_dim=8)(gt[None, :])
        vecs.append(enc["reg_target"].numpy()[enc["cls_target"].numpy() == 1][0])
    print(f"   20 deg -> sin={vecs[0][6]:+.5f} cos={vecs[0][7]:+.5f}", flush=True)
    print(f"  160 deg -> sin={vecs[1][6]:+.5f} cos={vecs[1][7]:+.5f}", flush=True)
    _check("cos channel is actually written (not left at zero)",
           abs(vecs[0][7] - np.cos(np.deg2rad(20.0))) < 1e-5)
    _check("sin agrees for 20 and 160 (why asin cannot separate them)",
           abs(vecs[0][6] - vecs[1][6]) < 1e-5)
    _check("cos separates 20 from 160 (why atan2 can)",
           abs(vecs[0][7] - vecs[1][7]) > 1e-3)

    model = _FakeModel(8, 8, 8, 8)
    got_dim = assert_reg_dim_consistent(model, _FakeDataset(8), CoRALoss(reg_dim=8))
    _check(f"assertion PASSES with everything at 8 (returned {got_dim})",
           got_dim == 8)


class _FakeModel:
    """Minimal stand-in: the assertion only reads .reg_dim off each component,
    so this exercises it without building a real CoRAModel."""

    def __init__(self, head, lc, pac, dec):
        self.local_head = type("H", (), {"reg_dim": head})()
        self.lc_head = type("H", (), {"reg_dim": lc})()
        self.pac = type("P", (), {"reg_dim": pac})()
        self.adaptive = type("A", (), {
            "decoder": type("D", (), {"reg_dim": dec})()})()


class _FakeDataset:
    def __init__(self, reg_dim):
        self.target_assigner = type("T", (), {"reg_dim": reg_dim})()


def case_c() -> None:
    print("\n=== CASE C  mismatch -- the assertion must FIRE ===", flush=True)
    scenarios = [
        ("head=8 loss=7", _FakeModel(8, 8, 8, 8), _FakeDataset(8), 7),
        ("assigner=7 rest=8", _FakeModel(8, 8, 8, 8), _FakeDataset(7), 8),
        ("pac=7 rest=8", _FakeModel(8, 8, 7, 8), _FakeDataset(8), 8),
        ("decoder=7 rest=8", _FakeModel(8, 8, 8, 7), _FakeDataset(8), 8),
    ]
    for label, model, dataset, loss_dim in scenarios:
        try:
            assert_reg_dim_consistent(model, dataset, CoRALoss(reg_dim=loss_dim))
        except ValueError as exc:
            first = str(exc).splitlines()[0]
            _check(f"{label} -> fired: {first}", True)
        else:
            _check(f"{label} -> DID NOT FIRE", False)

    # The message must name every component and its value, or a 3am reader
    # cannot tell which side is wrong.
    try:
        assert_reg_dim_consistent(_FakeModel(8, 8, 8, 7), _FakeDataset(8),
                                  CoRALoss(reg_dim=8))
    except ValueError as exc:
        msg = str(exc)
        print("\n  full message for decoder=7:", flush=True)
        for line in msg.splitlines():
            print(f"    | {line}", flush=True)
        _check("message names every component",
               all(n in msg for n in ("local_head", "lc_head", "pac",
                                      "decoder", "loss", "assigner")))


def defaults() -> None:
    """The strictly-additive guarantee: nothing the other four packages call
    changed its default, and nothing was made required."""
    import inspect

    from cpbench.models.heads import DetectionHead
    from cpbench.training.losses import DetectionLoss

    print("\n=== NO DEFAULT MOVED ===", flush=True)
    for fn, name in ((DetectionHead.__init__, "DetectionHead"),
                     (TargetAssigner.__init__, "TargetAssigner"),
                     (BoxDecoder.__init__, "BoxDecoder"),
                     (DetectionLoss.__init__, "DetectionLoss"),
                     (CoRALoss.__init__, "CoRALoss")):
        p = inspect.signature(fn).parameters["reg_dim"]
        print(f"  {name:<16} reg_dim default = {p.default!r}", flush=True)
        _check(f"{name}.reg_dim still defaults to 7", p.default == 7)


def main() -> int:
    case_a()
    case_b()
    case_c()
    defaults()
    print("", flush=True)
    if _FAILURES:
        print(f"REPORT: FAIL -- {len(_FAILURES)} check(s) failed:", flush=True)
        for f in _FAILURES:
            print(f"  - {f}", flush=True)
        return 1
    print("REPORT: PASS -- all checks green", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
