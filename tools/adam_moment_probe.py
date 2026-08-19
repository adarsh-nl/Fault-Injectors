"""Zero-GPU coarse precedence check from saved checkpoints (sec 7.14).

Adam's exp_avg_sq is an EMA of grad^2 with beta2=0.999 (~1000-step
horizon). Checkpoints at steps 2000/4000/6000/7667 of job 561546 therefore
carry four snapshots of smoothed per-parameter gradient magnitude across
the approach to the loss-trend abort. This cannot resolve ORDERING inside
a window (that is what the per-step probe is for) -- it can only say
whether the selective groups' gradient magnitude grew between snapshots,
and relative to upstream/downstream reference groups.

Param-index mapping: Adam state is keyed by position in model.parameters().
The teacher is built LAZILY at first forward, AFTER the optimizer was
constructed, so teacher params are absent from Adam state while present in
the model state dict. We therefore walk the model state-dict keys in
order, skipping 'teacher.'-prefixed keys and name-identified buffers, and
require an exact shape match against the Adam slot at every position --
any mismatch aborts loudly rather than misattributing a norm.
"""
import sys

import torch

sys.path.insert(0, "/home/nanjaiyalathaa/Fault-Injectors")
from corabench.compat import load as torch_load  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731

BUFFER_MARKERS = ("running_mean", "running_var", "num_batches_tracked")

GROUPS = {
    "a_log": lambda n: n == "lc.cssm.scan.a_log",
    "dt_bias": lambda n: n == "lc.cssm.scan.dt_bias",
    "dt_proj": lambda n: n.startswith("lc.cssm.dt_proj."),
    "b_proj": lambda n: n.startswith("lc.cssm.b_proj."),
    "c_proj": lambda n: n.startswith("lc.cssm.c_proj."),
    "out_norm": lambda n: n.startswith("lc.cssm.out_norm."),
    "gate_out": lambda n: n.startswith("lc.gate.out."),
    # references OUTSIDE the CSSM path
    "REF_encoder": lambda n: n.startswith("encoder."),
    "REF_lc_branches": lambda n: n.startswith(("lc.branch_", "lc.att.")),
    "REF_heads": lambda n: ("head" in n) and not n.startswith("teacher."),
}


def analyse(path):
    ck = torch_load(path, map_location="cpu")
    args = ck.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)
    step = ck.get("step")
    names = []
    for k in ck["model"]:
        if k.startswith("teacher.") or any(m in k for m in BUFFER_MARKERS):
            continue
        names.append(k)
    st = ck["opt"]["state"]
    idxs = sorted(st.keys())
    if len(idxs) != len(names):
        P("FATAL %s: %d adam slots vs %d candidate params -- mapping "
          "ambiguous, refusing to guess" % (path, len(idxs), len(names)))
        return None, step, args
    out = {}
    for i, n in zip(idxs, names):
        v = st[i].get("exp_avg_sq")
        if v is None or tuple(v.shape) != tuple(ck["model"][n].shape):
            P("FATAL %s: shape mismatch at slot %d (%s) -- %s vs %s"
              % (path, i, n, tuple(v.shape) if v is not None else None,
                 tuple(ck["model"][n].shape)))
            return None, step, args
        for g, pred in GROUPS.items():
            if pred(n):
                out.setdefault(g, []).append(float(v.sum()))
    # RMS of the EMA'd gradient: sqrt(sum(exp_avg_sq)/count) per group
    res = {}
    for g, sums in out.items():
        res[g] = sum(sums) ** 0.5      # l2 norm of sqrt(exp_avg_sq) proxy
    return res, step, args


if __name__ == "__main__":
    paths = sys.argv[1:]
    rows = []
    for p in paths:
        res, step, args = analyse(p)
        P("%s: step=%s seed=%s" % (p, step, args.get("seed")))
        if res:
            rows.append((step, res))
    if rows:
        groups = sorted(rows[0][1].keys())
        P("\nsqrt(sum exp_avg_sq) per group (EMA-of-grad^2 magnitude proxy),"
          "\nand its ratio to the FIRST snapshot:")
        P("%-14s" % "group" + "".join("%16s" % ("step %s" % s)
                                      for s, _ in rows))
        for g in groups:
            base = rows[0][1][g]
            P("%-14s" % g + "".join(
                "%16s" % ("%.3e (%5.1fx)" % (r[g], r[g] / (base + 1e-30)))
                for _, r in rows))
