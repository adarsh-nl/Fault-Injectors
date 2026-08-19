"""Print the three CoSDH gate verdicts from a results directory."""

import json
import os
import sys

PAPER = {"ap_50": 0.9683, "ap_70": 0.9299}
FLOOR = 1e-4


def load(d, *parts):
    p = os.path.join(d, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def main(root):
    g1 = load(root, "gate1_unwrapped", "fi_result.json")
    g2 = load(root, "gate2_wrapped_null", "fi_result.json")
    g3 = load(root, "gate3_pose", "gate3.json")

    print("=" * 72)
    print("GATE 1  clean baseline, UNWRAPPED")
    print("=" * 72)
    if not g1:
        print("  MISSING")
    else:
        print("  AP@0.5 = %.4f   paper %.4f   delta %+.4f"
              % (g1.get("ap_50", float("nan")), PAPER["ap_50"],
                 g1.get("ap_50", float("nan")) - PAPER["ap_50"]))
        print("  AP@0.7 = %.4f   paper %.4f   delta %+.4f"
              % (g1.get("ap_70", float("nan")), PAPER["ap_70"],
                 g1.get("ap_70", float("nan")) - PAPER["ap_70"]))
        print("  AP@0.3 = %.4f   n_gt = %s" % (g1.get("ap_30", float("nan")),
                                               g1.get("n_gt")))
        print("  NOTE the release README states the published checkpoints were")
        print("       RETRAINED and differ from the paper, so a delta here is")
        print("       expected and is NOT by itself a failure.")

    print()
    print("=" * 72)
    print("GATE 2  wrapped-null, same session/node as gate 1")
    print("=" * 72)
    if not (g1 and g2):
        print("  MISSING")
    else:
        d50 = abs(g2.get("ap_50", 0) - g1.get("ap_50", 0))
        d70 = abs(g2.get("ap_70", 0) - g1.get("ap_70", 0))
        print("  unwrapped    ap50=%.9f ap70=%.9f" % (g1["ap_50"], g1["ap_70"]))
        print("  wrapped-null ap50=%.9f ap70=%.9f" % (g2["ap_50"], g2["ap_70"]))
        print("  |delta|      d50=%.3e d70=%.3e   floor %.0e" % (d50, d70, FLOOR))
        print("  VERDICT: %s" % ("PASS" if (d50 < FLOOR and d70 < FLOOR)
                                 else "FAIL -- the wrapper alters the input"))

    print()
    print("=" * 72)
    print("GATE 3  pose fire-check + magnitude/spread/clean-slot/yaw")
    print("=" * 72)
    if not g3:
        print("  MISSING")
    else:
        s = g3["summary"]
        a, b, d = s["A_magnitude"], s["B_per_frame_spread"], s["D_yaw"]
        print("  frames=%d  pairs=%d" % (s["n_frames"], s["n_pairs"]))
        print("  A magnitude   pose_xy %.4f | transformation_matrix %.4f | "
              "pairwise_world %.4f" % (a["mean_d_pose_xy"],
                                       a["mean_d_transformation_matrix"],
                                       a["mean_d_pairwise_world"]))
        print("                theory (Rayleigh mean, sigma=%.2f) = %.4f"
              % (a["sigma_xy"], a["theory_rayleigh_mean"]))
        print("  B spread      unique=%d  std=%.4f  [%.4f, %.4f]  %s"
              % (b["unique_d_pose_xy"], b["std_d_pose_xy"], b["min"], b["max"],
                 "IDENTICAL ACROSS FRAMES -- BUG"
                 if b["IDENTICAL_ACROSS_FRAMES"] else "varies per frame OK"))
        print("  C clean slot  %s" % ("untouched OK" if s["C_clean_slot_untouched"]
                                      else "MUTATED -- fault would be inert"))
        print("  D yaw         pose d_yaw std=%.4f deg | matrix |d_yaw| mean="
              "%.4f deg | sigma=%.2f"
              % (d["std_d_yaw_pose_deg"], d["mean_abs_d_yaw_matrix_deg"],
                 d["sigma_yaw_deg"]))
        print("                slots touched: %s (expect [0, 1, 4])"
              % d["slots_touched"])
        ok = (not b["IDENTICAL_ACROSS_FRAMES"] and s["C_clean_slot_untouched"]
              and d["slots_touched"] == [0, 1, 4]
              and a["mean_d_transformation_matrix"] > 0
              and a["mean_d_pairwise_world"] > 0)
        print("  VERDICT: %s" % ("PASS" if ok else "INSPECT -- see above"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
