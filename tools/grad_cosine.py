"""Gradient-cosine experiment (spec sec 7.15). No training required.

WHY: rung 2 (--no_pac --no_teacher) DIVERGES on both seeds; rung 3
(--no_teacher, PAC on) is clean on both. PAC consumes the LOCAL branch's
outputs, not LC's, so PAC cannot be shielding LC through the forward path.
The remaining channel is the SHARED TRUNK: L_LC, L_PAC and L_local all
backpropagate into the same encoder/LC parameters, so removing L_PAC
changes the resultant gradient those parameters see.

WHAT: load a checkpoint, take ONE fixed batch, and backward each loss term
SEPARATELY (retain_graph) to get g_LC, g_PAC, g_local restricted to the
parameters shared upstream of all three paths. Report norms, all pairwise
cosines, and ||g_LC|| / ||g_LC + g_PAC + g_local||.

Registered interpretation is in docs/cora_spec.md sec 7.15 and was written
BEFORE this ran.

Runs on ONE batch on a GPU; submitted via sbatch, never on the login node.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "/home/nanjaiyalathaa/Fault-Injectors")

from corabench.compat import load as torch_load          # noqa: E402
from corabench.models.cora import CoRAModel              # noqa: E402
from corabench.scripts.train_opencood import build, epoch_loader  # noqa: E402
from corabench.training.losses import CoRALoss           # noqa: E402

P = lambda *a: print(*a, flush=True)                     # noqa: E731


def shared_params(model):
    """Parameters UPSTREAM of all three heads, i.e. everything the three
    losses genuinely share. Excludes each head's own parameters, whose
    gradients are trivially disjoint and would dilute every cosine toward 0
    by construction."""
    out = []
    for n, p in model.named_parameters():
        if n.startswith("teacher."):
            continue
        if not p.requires_grad:
            continue
        # head-specific params are NOT shared
        if any(k in n for k in ("head_local", "head_lc", "head_pac",
                                "recal", "u_lc", "u_pac")):
            continue
        out.append((n, p))
    return out


def grad_of(term, named, retain=True):
    for _, p in named:
        if p.grad is not None:
            p.grad = None
    term.backward(retain_graph=retain)
    return torch.cat([(p.grad.detach().float().flatten()
                       if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device))
                      for _, p in named])


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def run_ckpt(path, args, batch):
    P("=" * 78)
    P("CHECKPOINT %s" % path)
    P("=" * 78)
    ck = torch_load(path, map_location="cpu")
    step = ck.get("step")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoRAModel(args._grid_hw, channels=256, reg_dim=7,
                      head_hw=args._head_hw, dt_bound=0.2,
                      fp32_island=True, pac_enabled=True,
                      teacher_enabled=False).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [m for m in missing if not m.startswith("teacher.")]
    unexpected = [u for u in unexpected if not u.startswith("teacher.")]
    P("step=%s | load: %d missing, %d unexpected (teacher keys ignored)"
      % (step, len(missing), len(unexpected)))
    if missing[:3] or unexpected[:3]:
        P("  missing[:3]=%s unexpected[:3]=%s" % (missing[:3], unexpected[:3]))
    model.train()
    named = shared_params(model)
    ntot = sum(p.numel() for _, p in named)
    P("shared upstream params: %d tensors, %d scalars" % (len(named), ntot))

    loss_fn = CoRALoss(reg_dim=7, lambda_align=1.0).to(device)
    b = {k: (v.to(device) if torch.is_tensor(v) else v)
         for k, v in batch.items()}
    out = model(b)
    parts = loss_fn(out, b)

    # the three DETECTION terms, each the sum of its own cls+reg, matching
    # exactly how CoRALoss weights them (w_local = w_lc = w_pac = 1.0).
    terms = {
        "local": parts["loss_local_cls"] + 2.0 * parts["loss_local_reg"],
        "LC":    parts["loss_lc_cls"] + 2.0 * parts["loss_lc_reg"],
        "PAC":   parts["loss_pac_cls"] + 2.0 * parts["loss_pac_reg"],
    }
    g = {}
    for i, (k, t) in enumerate(terms.items()):
        g[k] = grad_of(t, named, retain=(i < len(terms) - 1))
        P("  ||g_%-6s|| = %.6e   (loss term %.4f)" % (k, float(g[k].norm()),
                                                      float(t)))
    tot = g["LC"] + g["PAC"] + g["local"]
    P("  ||g_LC + g_PAC + g_local|| = %.6e" % float(tot.norm()))
    P("")
    P("  cos(g_LC,    g_PAC)   = %+.6f" % cos(g["LC"], g["PAC"]))
    P("  cos(g_LC,    g_local) = %+.6f" % cos(g["LC"], g["local"]))
    P("  cos(g_PAC,   g_local) = %+.6f" % cos(g["PAC"], g["local"]))
    P("")
    P("  ||g_LC|| / ||sum||    = %.6f" % (float(g["LC"].norm())
                                          / (float(tot.norm()) + 1e-30)))
    P("  ||g_PAC||/ ||sum||    = %.6f" % (float(g["PAC"].norm())
                                          / (float(tot.norm()) + 1e-30)))
    P("  ||g_local||/||sum||   = %.6f" % (float(g["local"].norm())
                                          / (float(tot.norm()) + 1e-30)))
    # counterfactual: what rung 2 actually optimises (LC + local, no PAC)
    nopac = g["LC"] + g["local"]
    P("")
    P("  RUNG-2 COUNTERFACTUAL (LC + local, PAC removed):")
    P("    ||g_LC + g_local||   = %.6e" % float(nopac.norm()))
    P("    ||g_LC|| / ||LC+local|| = %.6f"
      % (float(g["LC"].norm()) / (float(nopac.norm()) + 1e-30)))
    P("    cos(g_LC, g_LC+g_local) = %+.6f" % cos(g["LC"], nopac))
    P("    cos(sum_all, LC+local)  = %+.6f" % cos(tot, nopac))
    P("")
    del model, out, parts, g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypes", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2029)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    adapter, grid_hw = build(args.hypes, train=True)
    args._grid_hw = grid_hw
    args._head_hw = tuple(adapter[0]["targets"]["cls_target"].shape[:2])
    P("dataset %d samples | canvas %s | head %s"
      % (len(adapter), grid_hw, args._head_hw))
    # ONE fixed batch, identical for every checkpoint: the comparison is
    # across checkpoints, so the batch must not vary.
    loader = epoch_loader(adapter, args, 0)
    batch = next(iter(loader))
    P("batch keys: %s" % sorted(k for k in batch if torch.is_tensor(batch[k])))
    for c in args.ckpt:
        try:
            run_ckpt(c, args, batch)
        except Exception as e:                              # noqa: BLE001
            P("FAILED on %s: %s: %s" % (c, type(e).__name__, str(e)[:500]))
            import traceback
            traceback.print_exc()
    P("GRAD-COSINE DONE")
