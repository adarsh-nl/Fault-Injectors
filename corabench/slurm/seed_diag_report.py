"""Answer the three questions the 5-seed basin diagnostic was run to settle.

Usage: python seed_diag_report.py <results/cora_seeddiag/<jobid>>

  Q1  how many seeds enter the bad basin, and at what step
  Q2  does LC activation magnitude diverge BEFORE the first non-finite value
  Q3  do failing seeds differ from surviving seeds in their OPENING skip
      pattern, and does basin entry track early GradScaler backoffs

Deliberately does NOT test delta saturation as a cause: within-window it does
not separate non-finite from finite steps (job 559279), so delta is reported
as a co-symptom column only.
"""
import csv
import glob
import math
import os
import sys

# basin = a sustained non-finite regime, not one unlucky step. 559279 reached
# 8/25 non-finite by step 50-74 and never recovered; 559057 had 9 skips in
# 5000 steps and no non-finite loss at all.
WIN = 50
FRAC = 0.20


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float('nan')


def nan(x):
    return x != x


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, 'seed*', '*.csv'))):
        seed = os.path.basename(os.path.dirname(p)).replace('seed', '')
        out[seed] = list(csv.DictReader(open(p)))
    return out


def basin_step(rows):
    """first step at which the trailing WIN window is >= FRAC non-finite."""
    flags = [nan(num(r, 'loss')) for r in rows]
    for i in range(len(flags)):
        lo = max(0, i - WIN + 1)
        w = flags[lo:i + 1]
        if len(w) >= WIN and sum(w) / len(w) >= FRAC:
            return int(rows[i]['step'])
    return None


def first_nonfinite(rows):
    for r in rows:
        if nan(num(r, 'loss')):
            return int(r['step'])
    return None


def main(d):
    runs = load(d)
    if not runs:
        print('no seed CSVs under', d)
        return 1

    print('=' * 78)
    print('Q1  BASIN ENTRY  (window %d, threshold %.0f%% non-finite)'
          % (WIN, 100 * FRAC))
    print('=' * 78)
    print('%-8s %-9s %-11s %-13s %-9s %s'
          % ('seed', 'steps', 'basin@', 'first-nonfin', 'nonfin%', 'verdict'))
    bad, good = [], []
    for s, rows in sorted(runs.items()):
        b = basin_step(rows)
        f = first_nonfinite(rows)
        nf = sum(1 for r in rows if nan(num(r, 'loss')))
        (bad if b is not None else good).append(s)
        print('%-8s %-9d %-11s %-13s %-9.1f %s'
              % (s, len(rows), b if b is not None else '-',
                 f if f is not None else '-', 100.0 * nf / max(1, len(rows)),
                 'BAD BASIN' if b is not None else 'survived'))
    print('\n  %d of %d seeds entered the bad basin  (bad=%s survived=%s)'
          % (len(bad), len(runs), ','.join(bad) or '-', ','.join(good) or '-'))

    print()
    print('=' * 78)
    print('Q2  IS THERE AN EARLY WARNING?  LC/CSSM magnitude before the first')
    print('    non-finite value.  If the magnitude ramps ahead of the NaN, a')
    print('    cheap runtime breaker is possible; if it jumps only at the NaN')
    print('    step, there is nothing to trip on.')
    print('=' * 78)
    cols = [c for c in ('lc_in_ego_max', 'cssm_in_max', 'cssm_in_p99',
                        'cssm_out_max', 'lc_out_max')]
    for s, rows in sorted(runs.items()):
        f = first_nonfinite(rows)
        print('\n seed %s  (first non-finite: %s)' % (s, f if f else 'none'))
        probed = [r for r in rows if not nan(num(r, 'cssm_in_max'))]
        if not probed:
            print('   no activation probe columns')
            continue
        if f is None:
            base = probed[:20]
            late = probed[-20:]
            for c in cols:
                b = [num(r, c) for r in base if not nan(num(r, c))]
                l = [num(r, c) for r in late if not nan(num(r, c))]
                if b and l:
                    print('   %-14s first20 mean=%-11.4f last20 mean=%-11.4f '
                          'growth x%.2f'
                          % (c, sum(b) / len(b), sum(l) / len(l),
                             (sum(l) / len(l)) / max(1e-12, sum(b) / len(b))))
            continue
        pre = [r for r in probed if int(r['step']) < f]
        if len(pre) < 4:
            print('   fewer than 4 probed steps before the NaN -- cannot tell')
            continue
        k = max(1, len(pre) // 4)
        for c in cols:
            e = [num(r, c) for r in pre[:k] if not nan(num(r, c))]
            l = [num(r, c) for r in pre[-k:] if not nan(num(r, c))]
            if e and l:
                em, lm = sum(e) / len(e), sum(l) / len(l)
                print('   %-14s early=%-11.4f just-before-NaN=%-11.4f  x%.2f%s'
                      % (c, em, lm, lm / max(1e-12, em),
                         '   <-- RAMP' if lm > 3 * em else ''))

    print()
    print('=' * 78)
    print('Q3  OPENING SKIP PATTERN + GRADSCALER.  All runs are bit-identical')
    print('    through step 3 and diverge at step 4 on a skip/no-skip flip.')
    print('=' * 78)
    print('%-8s %-9s %-12s %-12s %-14s %s'
          % ('seed', 'basin', 'skips[0:10]', 'skips[0:50]', 'scale@50',
             'first 12 stepped'))
    for s, rows in sorted(runs.items()):
        b = basin_step(rows)
        sk = [r.get('stepped') for r in rows]
        s10 = sum(1 for x in sk[:10] if x == '0')
        s50 = sum(1 for x in sk[:50] if x == '0')
        sc = next((num(r, 'scale') for r in rows if int(r['step']) == 50),
                  float('nan'))
        print('%-8s %-9s %-12d %-12d %-14.0f %s'
              % (s, 'BAD' if b is not None else 'ok', s10, s50, sc,
                 ''.join(x if x in ('0', '1') else '?' for x in sk[:12])))

    bad_s = [s for s in runs if basin_step(runs[s]) is not None]
    good_s = [s for s in runs if basin_step(runs[s]) is None]
    if bad_s and good_s:
        def avg(ss, n):
            v = [sum(1 for x in [r.get('stepped') for r in runs[s]][:n]
                     if x == '0') for s in ss]
            return sum(v) / len(v)
        print('\n  mean skips in first 10 steps:  bad=%.2f  survived=%.2f'
              % (avg(bad_s, 10), avg(good_s, 10)))
        print('  mean skips in first 50 steps:  bad=%.2f  survived=%.2f'
              % (avg(bad_s, 50), avg(good_s, 50)))
        print('\n  If the bad seeds back off more in the opening steps, an\n'
              '  init_scale reduction or a scale warmup is a MITIGATION THAT\n'
              '  TOUCHES NO MODEL CODE (--init_scale is already wired), and\n'
              '  should be tried before anything architectural.')
    else:
        print('\n  all seeds fell on the same side -- opening-pattern contrast\n'
              '  is not measurable from this run; rerun with more seeds.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
