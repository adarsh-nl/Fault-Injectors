"""Proof-run verdict: report the observables the previous gates never read.

Usage: python proof_verdict.py <train_loss.csv> [dt_bound]

``dt_bound`` defaults to 0.2 and MUST match the value the run was launched
with. It was hardcoded at 0.2 until 2026-08-08, which made the dt_max=0.3 A/B
(job 559278) report INSPECT purely because 0.3 > 0.2001 -- a false alarm about
a healthy run. The verdict is diagnostic only; the sbatch gates
``--dependency=afterok`` on the TRAINER's exit code, not on this script, so
the false INSPECT did not block the chained full run.
"""
import csv
import math
import sys


def g(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float('nan')


def main(path, bound=0.2):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print('no rows logged')
        return 1
    nf = [r for r in rows if math.isnan(g(r, 'loss'))]
    print('steps run: %d    NON-FINITE-loss steps: %d (%.2f%%)'
          % (len(rows), len(nf), 100.0 * len(nf) / len(rows)))
    print()
    print('%-7s %-11s %-11s %-11s %-10s %-10s %s'
          % ('step', 'delta_max', 'delta_p99', 'delta_mean', 'sat_frac',
             't/s ratio', 'loss'))
    for s in (0, 100, 500, 1000, 2000, 3000, 3400, 3501, 3600, 4000, 4500,
              len(rows) - 1):
        m = [r for r in rows if int(r['step']) == s]
        if not m:
            continue
        r = m[0]
        tr = g(r, 't_ratio')
        print('%-7s %-11.5f %-11.5f %-11.5f %-10.5f %-10s %s'
              % (s, g(r, 'delta_max'), g(r, 'delta_p99'), g(r, 'delta_mean'),
                 g(r, 'sat_frac'),
                 ('%.3f' % tr) if not math.isnan(tr) else 'n/a', r['loss']))

    def col(k):
        return [g(r, k) for r in rows if not math.isnan(g(r, k))]
    dm, sf, tr = col('delta_max'), col('sat_frac'), col('t_ratio')
    tmax = col('t_teacher_absmax')
    print()
    print('WHOLE RUN:')
    print('  delta_max        max=%.5f   (bound %.3g, abort at 1.0)'
          % ((max(dm) if dm else float('nan')), bound))
    print('  sat_frac         max=%.4f   mean=%.4f'
          % (max(sf) if sf else float('nan'),
             (sum(sf) / len(sf)) if sf else float('nan')))
    print('  teacher/student  max=%.3f   (warn at 4.0)'
          % (max(tr) if tr else float('nan')))
    print('  teacher |f|max   max=%.1f   (abort at 16376)'
          % (max(tmax) if tmax else float('nan')))
    # loss trend, first vs last 200 finite steps
    fin = [g(r, 'loss') for r in rows if not math.isnan(g(r, 'loss'))]
    if len(fin) > 400:
        a = sum(fin[:200]) / 200
        b = sum(fin[-200:]) / 200
        print('  loss             first200=%.4f -> last200=%.4f  (%s)'
              % (a, b, 'descending' if b < a else 'NOT descending'))
    ok = (dm and max(dm) <= bound * 1.0005) and not nf and len(rows) > 4000
    print()
    print('PROOF VERDICT: %s'
          % ('PASS -- delta bounded, zero non-finite loss, ran past step 4000'
             if ok else 'INSPECT -- see numbers above'))
    if not ok and dm and max(dm) <= bound * 1.0005:
        print('  (delta IS within the bound; the INSPECT is from the '
              'non-finite-loss or step-count clause, not from delta)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1],
                  float(sys.argv[2]) if len(sys.argv) > 2 else 0.2))
