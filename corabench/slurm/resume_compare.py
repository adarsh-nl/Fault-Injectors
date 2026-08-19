"""Compare an uninterrupted 2N-step run against N + resume + N.

Reports what matches EXACTLY and what only approximately, with numbers.
A plausible-looking loss curve is not evidence; the discriminators are lr
(must be exact -- it is a pure function of scheduler position), scaler scale
(exact if the scaler state restored), and the loss trajectory (approximate at
best under GPU nondeterminism).

Usage: resume_compare.py A.csv B1.csv B2.csv N
"""
import csv
import torch
from cpbench.utils.torchio import load as _torch_load
import math
import os
import sys


def load(p):
    with open(p) as fh:
        return {int(r["step"]): r for r in csv.DictReader(fh)}


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def main(pa, pb1, pb2, n, ckpt=None):
    A, B1, B2 = load(pa), load(pb1), load(pb2)
    n = int(n)
    print("run A steps: %d   run B1: %d   run B2: %d"
          % (len(A), len(B1), len(B2)))

    # ---- 1. pre-checkpoint segment: B1 vs A, steps 0..n-1 ----------------
    print("\n1. PRE-CHECKPOINT (steps 0..%d): B1 vs A" % (n - 1))
    common = sorted(set(A) & set(B1))
    for key, tol in (("lr", 0.0), ("scale", 0.0), ("loss", None)):
        va = [f(A[s], key) for s in common]
        vb = [f(B1[s], key) for s in common]
        exact = all((x == y) or (math.isnan(x) and math.isnan(y))
                    for x, y in zip(va, vb))
        if key == "loss":
            d = [abs(x - y) for x, y in zip(va, vb)
                 if not (math.isnan(x) or math.isnan(y))]
            print("   %-6s exact=%-5s  max|Δ|=%.3e  mean|Δ|=%.3e"
                  % (key, exact, max(d) if d else float("nan"),
                     (sum(d) / len(d)) if d else float("nan")))
        else:
            print("   %-6s exact=%s" % (key, exact))

    # ---- 2. post-resume: LR is the ONLY valid A-vs-B2 discriminator ----
    # A and B1 are two FRESH runs and diverge by ~step 4 on a GradScaler
    # skip flip (GPU nondeterminism; the seed diagnostic measured this). So
    # B2, which continues B1, can NEVER match A on loss or scaler. LR can:
    # it is a pure function of scheduler position, independent of any of
    # that. Comparing scaler A-vs-B2 was a design error in the first run.
    print("\n2. POST-RESUME (steps %d..%d): B2 vs A -- LR ONLY" % (n, 2 * n - 1))
    post = sorted(s for s in (set(A) & set(B2)) if s >= n)
    if not post:
        print("   NO OVERLAPPING POST-RESUME STEPS -- resume did not continue"
              " from the right step")
        return 1
    va = [f(A[s], "lr") for s in post]
    vb = [f(B2[s], "lr") for s in post]
    lr_ok = all(x == y for x, y in zip(va, vb))
    mism = [(s, x, y) for s, x, y in zip(post, va, vb) if x != y]
    print("   lr     exact=%-5s over %d steps %s"
          % (lr_ok, len(post), "" if lr_ok
             else "first mismatch step=%d A=%s B2=%s" % mism[0]))
    print("   (loss and scaler are NOT compared against A -- see above)")
    la = [f(A[s], "loss") for s in post]
    lb = [f(B2[s], "loss") for s in post]
    d = [abs(x - y) for x, y in zip(la, lb)
         if not (math.isnan(x) or math.isnan(y))]
    rel = [abs(x - y) / max(1e-9, abs(x)) for x, y in zip(la, lb)
           if not (math.isnan(x) or math.isnan(y))]
    print("   loss   max|Δ|=%.3e mean rel=%.3e  (INFORMATIONAL: bounded by "
          "A-vs-B1 pre-checkpoint divergence, not by resume)"
          % (max(d) if d else float("nan"),
             (sum(rel) / len(rel)) if rel else float("nan")))

    # ---- 3. scaler: B2's RESTORED state vs B1's SAVED state -------------
    print("\n3. SCALER: B2 restored vs B1's SAVED checkpoint state")
    sc_ok = None
    if ckpt and os.path.exists(ckpt):
        import torch
        saved = _torch_load(ckpt, map_location="cpu")
        want = (saved.get("scaler") or {}).get("scale")
        got = f(B2[min(post)], "scale") if post else float("nan")
        sc_ok = (want is not None and float(want) == got)
        print("   ckpt scaler scale = %s ; B2 first post-resume row = %s"
              % (want, got))
        print("   scaler exact=%s" % sc_ok)
    else:
        print("   checkpoint not supplied/found -- scaler NOT verified")

    # ---- 4. epoch boundary actually crossed? ----------------------------
    eps = sorted({int(B2[s].get("epoch", -1)) for s in post})
    crossed = len(eps) > 1
    print("\n4. EPOCH BOUNDARY: epochs seen post-resume = %s -> %s"
          % (eps, "CROSSED" if crossed else "NOT crossed (sched.step() and "
             "the per-epoch reseed were never exercised)"))

    print("\nVERDICT")
    print("  lr trajectory   %s  <- scheduler position restored"
          % ("EXACT" if lr_ok else "MISMATCH"))
    print("  scaler state    %s  <- vs B1's saved state"
          % ("EXACT" if sc_ok else ("MISMATCH" if sc_ok is False else "UNVERIFIED")))
    print("  epoch boundary  %s" % ("crossed" if crossed else "NOT crossed"))
    print("  RESUME %s" % ("VERIFIED" if (lr_ok and sc_ok and crossed)
                           else "NOT VERIFIED"))
    return 0 if (lr_ok and sc_ok and crossed) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
