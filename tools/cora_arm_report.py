"""Per-200-step loss decomposition for one sec-7.7 arm.

The discriminator is WHICH TERM MOVES, not survival, so this prints every
component on one axis with align's share of the total, and reports where the
run stopped relative to 561546's abort (7,667).
"""
import csv
import glob
import os
import statistics as st
import sys

KEYS = ("loss", "loss_align", "loss_local_cls", "loss_lc_cls", "loss_pac_cls",
        "loss_local_reg", "loss_lc_reg", "loss_pac_reg")
REF_ABORT = 7667          # 561546


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def main(root, arm):
    hits = glob.glob(os.path.join(root, "*.csv"))
    if not hits:
        print("no CSV under %s" % root)
        return 1
    rows = list(csv.DictReader(open(hits[0])))
    last = int(rows[-1]["step"])
    nf = sum(1 for r in rows if f(r, "loss") != f(r, "loss"))
    print("arm %s: %d rows, last step %d, non-finite %d (%.2f%%)"
          % (arm, len(rows), last, nf, 100.0 * nf / max(1, len(rows))))
    print("561546 aborted at %d -- this arm %s that point\n"
          % (REF_ABORT, "PASSED" if last > REF_ABORT else "did NOT reach"))

    print("%-6s %-9s %-9s %-8s %-8s %-8s %-8s %-8s %-8s %s"
          % ("win", "loss", "ALIGN", "loc_cls", "lc_cls", "pac_cls",
             "loc_reg", "lc_reg", "pac_reg", "align%"))
    for lo in range(0, last + 1, 200):
        b = [r for r in rows if lo <= int(r["step"]) < lo + 200]
        if not b:
            continue
        m = {}
        for k in KEYS:
            v = [f(r, k) for r in b if f(r, k) == f(r, k)]
            m[k] = st.mean(v) if v else float("nan")
        share = (100.0 * m["loss_align"] / m["loss"]
                 if m["loss"] == m["loss"] and m["loss"] else float("nan"))
        print("%-6d %-9.3f %-9.4f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %.2f%%"
              % (lo, m["loss"], m["loss_align"], m["loss_local_cls"],
                 m["loss_lc_cls"], m["loss_pac_cls"], m["loss_local_reg"],
                 m["loss_lc_reg"], m["loss_pac_reg"], share))

    fin = [f(r, "loss") for r in rows if f(r, "loss") == f(r, "loss")]
    print("\nloss first200=%.3f last200=%.3f   align max share=%.2f%%"
          % (st.mean(fin[:200]), st.mean(fin[-200:]),
             max(100.0 * f(r, "loss_align") / f(r, "loss") for r in rows
                 if f(r, "loss") == f(r, "loss") and f(r, "loss") > 0
                 and f(r, "loss_align") == f(r, "loss_align"))))
    print("REMINDER: 9,000 steps does NOT clear 559640's turn (~16k). "
          "Survival here bounds the FAST mode only.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "?"))
