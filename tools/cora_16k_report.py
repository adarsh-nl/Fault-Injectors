"""Report CoRA 561546 at step 16,000 -- the point where 559640 turned.

559640 looked healthy at step 4,720 and had already turned by 16k-18k, so a
pass/fail at 16k is not enough. This reports the three quantities that
actually moved on 559640, each against its own history rather than a
threshold:

  * loss_align          -- rose 0.005 -> 3.859 per-block on 559640
  * t_student_absmax    -- rose 1.17 -> 17.69 per-block, unwatched at the time
  * trailing-200 mean loss vs its RUNNING MINIMUM -- the loss-trend breaker's
    own statistic, so the margin to abort is visible before it fires
"""
import csv
import glob
import os
import statistics as st
import sys
from collections import deque

CSV = "results/cora_full/561546/train_loss.csv"
REF = "results/cora_full/559640/train_loss.csv"      # the run that diverged


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def blocks(rows, key, size=2000):
    out = []
    hi = max(int(r["step"]) for r in rows)
    for lo in range(0, hi + 1, size):
        v = [f(r, key) for r in rows
             if lo <= int(r["step"]) < lo + size and f(r, key) == f(r, key)]
        if v:
            out.append((lo, st.mean(v)))
    return out


def trend(rows, window=200):
    """trailing-window mean and its running minimum, the breaker's statistic."""
    w, lmin, cur = deque(maxlen=window), None, float("nan")
    for r in rows:
        v = f(r, "loss")
        if v != v:
            continue
        w.append(v)
        if len(w) == window:
            cur = sum(w) / len(w)
            if lmin is None or cur < lmin:
                lmin = cur
    return cur, lmin


def main():
    if not os.path.exists(CSV):
        print("no CSV at %s" % CSV)
        return 1
    rows = list(csv.DictReader(open(CSV)))
    step = int(rows[-1]["step"])
    nf = sum(1 for r in rows if f(r, "loss") != f(r, "loss"))
    rate_a, rate_b = rows[10], rows[-1]
    rate = ((f(rate_b, "sec") - f(rate_a, "sec"))
            / (int(rate_b["step"]) - int(rate_a["step"])))
    spe = 3382
    print("561546 @ step %d (epoch %.2f of 30) | %.3f s/step | non-finite %d/%d"
          % (step, step / spe, rate, nf, len(rows)))

    cur, lmin = trend(rows)
    print("\nLOSS-TREND (the breaker's own statistic, aborts at 5x)")
    print("  trailing-200 mean = %.4f   running min = %.4f   ratio = %.3fx"
          % (cur, lmin if lmin else float("nan"),
             (cur / lmin) if lmin else float("nan")))

    print("\nPER-2000-STEP BLOCKS   561546            559640 (diverged)")
    ref = list(csv.DictReader(open(REF))) if os.path.exists(REF) else []
    for key in ("loss", "loss_align", "t_student_absmax"):
        a = dict(blocks(rows, key))
        b = dict(blocks(ref, key)) if ref else {}
        print("  %s" % key)
        for lo in sorted(a):
            print("    %5d-%5d  %10.4f       %s"
                  % (lo, lo + 1999, a[lo],
                     ("%10.4f" % b[lo]) if lo in b else "-"))

    print("\nLAST-100 COMPONENTS")
    for k in ("loss_local_cls", "loss_lc_cls", "loss_pac_cls", "loss_align",
              "t_student_absmax", "t_teacher_absmax", "t_ratio"):
        v = [f(r, k) for r in rows[-100:] if f(r, k) == f(r, k)]
        if v:
            print("  %-18s %.4f" % (k, sum(v) / len(v)))

    ck = sorted(glob.glob("results/cora_full/561546/ckpt_step*.pt"))
    print("\ncheckpoints: %d  %s" % (len(ck), " ".join(os.path.basename(c)
                                                       for c in ck[-4:])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
