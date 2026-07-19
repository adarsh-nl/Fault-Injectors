"""Loss functions: finiteness, positivity, sane orderings, gradients."""

import torch

from corabench.training.losses import (CoRALoss, focal_loss_prob,
                                       smooth_l1_reg_loss)


def _targets(h=6, w=6, a=2, positives=((1, 2, 0),)):
    cls_t = torch.zeros(1, h, w, a)
    reg_t = torch.zeros(1, h, w, a, 7)
    for (i, j, k) in positives:
        cls_t[0, i, j, k] = 1.0
        reg_t[0, i, j, k] = 0.5
    return cls_t, reg_t


def test_focal_loss_prefers_correct_predictions():
    cls_t, _ = _targets()
    good = torch.full((1, 2, 6, 6), 0.02)
    good[0, 0, 1, 2] = 0.98
    bad = torch.full((1, 2, 6, 6), 0.98)
    assert focal_loss_prob(good, cls_t) < focal_loss_prob(bad, cls_t)


def test_focal_loss_ignores_ignore_label():
    cls_t, _ = _targets()
    cls_t[0, 3, 3, 1] = -1.0
    base = focal_loss_prob(torch.full((1, 2, 6, 6), 0.5), cls_t)
    # perturbing the ignored cell's prediction must not change the loss
    pred = torch.full((1, 2, 6, 6), 0.5)
    pred[0, 1, 3, 3] = 0.99
    assert torch.allclose(base, focal_loss_prob(pred, cls_t))


def test_reg_loss_only_on_positives():
    cls_t, reg_t = _targets()
    reg_map = torch.zeros(1, 14, 6, 6)
    loss0 = smooth_l1_reg_loss(reg_map, reg_t, cls_t)
    assert loss0 > 0                       # positive anchor target is 0.5
    exact = reg_t.permute(0, 3, 4, 1, 2).reshape(1, 14, 6, 6)
    assert smooth_l1_reg_loss(exact, reg_t, cls_t) < loss0


def test_cora_loss_composition(tiny_model, batch):
    tiny_model.train()
    out = tiny_model(batch, return_teacher=True)
    losses = CoRALoss()(out, batch)
    assert set(losses) == {"total", "cls", "reg", "align", "pac"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
