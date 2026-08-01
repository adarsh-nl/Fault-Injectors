#!/usr/bin/env python3
"""Read every cell in cells/ and regenerate RESULTS.md. Never hand-edit RESULTS.md.

Tables: 0 coverage/confounders, 1 headline, 2 per-fault sweeps, 3 agentdrop stratified.
Filled cell = mean ap70 +/- std (retention%) over seeds. Pending = '--'.
Retention is vs the run's own clean cell.
"""
import json, glob, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "table_spec.json")))
CELLS = os.path.join(HERE, "cells")

def load():
    idx = {}
    for fn in glob.glob(os.path.join(CELLS, "*.json")):
        r = json.load(open(fn))
        idx.setdefault((r["operator"], r["dataset"], r["fault"], r["severity"]), []).append(r)
    return idx

def ap70s(recs):
    return [r["metric"]["ap_70"] for r in recs]

def clean_ap(idx, op, ds):
    recs = idx.get((op, ds, "clean", "clean"), [])
    return st.mean(ap70s(recs)) if recs else None

def cell(idx, op, ds, fault, sev):
    recs = idx.get((op, ds, fault, sev), [])
    if not recs:
        return "--"
    vals = ap70s(recs)
    mean = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    base = clean_ap(idx, op, ds)
    ret = f" ({100*mean/base:.0f}%)" if base else ""
    return f"{mean:.3f}+/-{sd:.3f}{ret}"

def run_label(cov):
    return f"{cov['operator']} . {cov['dataset']}"

def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

def main():
    idx = load()
    cov = SPEC["coverage"]
    faults = SPEC["faults"]
    L = ["# Fault-Injection Results (generated, do not hand-edit)",
         "", "Cell = mean AP@0.7 +/- std over seeds, with (retention vs this run's clean).",
         "Seeds: " + ", ".join(map(str, SPEC["seeds"])) + ". '--' = pending.", ""]

    L += ["## Table 0. Coverage and protocol confounders", ""]
    rows = []
    for c in cov:
        p = c["protocol"]
        rows.append([run_label(c), c["domain"], f"comp {p['compression']}",
                     "noise-trained" if p["noise_trained"] else "clean-trained",
                     "bb-frozen" if p["backbone_fix"] else "bb-open",
                     f"{clean_ap(idx, c['operator'], c['dataset']) or '--'}"])
    L += [md_table(["Run", "Domain", "Compression", "Noise exposure",
                    "Backbone", "Clean AP@0.7"], rows), ""]

    L += ["## Table 1. Headline degradation (each fault at its headline severity)", ""]
    fault_names = list(faults.keys()) + ["agentdrop"]
    head = ["Run", "Clean"] + [f"{f} ({faults[f]['headline'] if f in faults else SPEC['agentdrop']['headline']})"
                               for f in fault_names]
    rows = []
    for c in cov:
        op, ds = c["operator"], c["dataset"]
        base = clean_ap(idx, op, ds)
        row = [run_label(c), f"{base:.3f}" if base else "--"]
        for f in fault_names:
            sev = faults[f]["headline"] if f in faults else SPEC["agentdrop"]["headline"]
            row.append(cell(idx, op, ds, f, sev))
        rows.append(row)
    L += [md_table(head, rows), ""]

    L += ["## Table 2. Full sweeps (one per fault)", ""]
    for f, meta in faults.items():
        L += [f"### {f}", ""]
        head = ["Run", "Clean"] + meta["severities"]
        rows = []
        for c in cov:
            op, ds = c["operator"], c["dataset"]
            base = clean_ap(idx, op, ds)
            row = [run_label(c), f"{base:.3f}" if base else "--"]
            row += [cell(idx, op, ds, f, s) for s in meta["severities"]]
            rows.append(row)
        L += [md_table(head, rows), ""]

    L += ["## Table 3. AgentDrop, stratified by agent count (never pooled)", ""]
    strata = SPEC["agentdrop"]["strata"]
    head = ["Run"] + [f"{n} agents" for n in strata]
    rows = []
    for c in cov:
        op, ds = c["operator"], c["dataset"]
        rows.append([run_label(c)] + [cell(idx, op, ds, "agentdrop", n) for n in strata])
    L += [md_table(head, rows), ""]

    out = os.path.join(HERE, "RESULTS.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
