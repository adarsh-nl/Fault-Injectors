"""Runtime training gate (job 558108 post-mortem, user-approved design).

The failure that motivated this: Delta escaped its Mamba range by step 2000,
the first NaN appeared at 3501, and the run then burned 22 more hours at a
>99% skip rate with nothing detecting it. Every observable needed to catch it
was inside the model the whole time and nothing was reading it.

The design constraint is therefore explicit: this must trip at roughly step
2000, where the pathology was observable, not at 3501 where it became fatal.

Three independent checks, evaluated every `interval` steps:

1. DELTA (the contract violation)
   delta_max > 2 x dt_bound      -> WARN
   delta_max > 5 x dt_bound      -> ABORT
   Also logs delta_p99, delta_mean and `sat_frac`, the fraction of channels
   with |delta / dt_bound| > 1. sat_frac is the measurement that settles
   whether a bounded run pushes channels into the tanh's saturating region;
   it is reported, not thresholded, because the question is open.

2. TEACHER / STUDENT MAGNITUDE (the leading indicator)
   The teacher is fed the DENSE unmasked collaborator sum and crossed the
   fp16 ceiling 825 steps before the student, so its magnitude -- not the
   loss -- is what moves first.
   ratio > 4                     -> WARN
   teacher_absmax > 0.25 * 65504 -> ABORT   (quarter-ceiling headroom)

3. NON-FINITE-LOSS SKIP RATE (the circuit breaker)
   >20% of the last 200 steps with a NON-FINITE LOSS -> ABORT.
   The loss-finiteness discriminator matters: a GradScaler backoff computes a
   FINITE loss and only overflows the scaled gradients, whereas a true forward
   failure makes the loss itself NaN. Measured on the two real runs:
       smoke 557653: 5 skips, 5 finite-loss (backoff), 0 NaN-loss
       full  558108: 51347 skips, 96 finite-loss, 51251 NaN-loss
   Perfect separation, so healthy openings are exempt by construction and no
   step floor is needed.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

FP16_MAX = 65504.0


class GateAbort(RuntimeError):
    """Raised when a gate trips at ABORT level; the trainer must stop."""


class RuntimeGate:
    def __init__(self, dt_bound: float = 0.2, interval: int = 100,
                 delta_warn_mult: float = 2.0, delta_abort_mult: float = 5.0,
                 ratio_warn: float = 4.0, ratio_warn_low: float = 0.25,
                 teacher_abort_frac: float = 0.25,
                 student_abort_frac: float = 0.25,
                 skip_window: int = 200, skip_abort_frac: float = 0.20,
                 loss_window: int = 200, loss_blowup_mult: float = 5.0,
                 loss_min_steps: int = 2000,
                 scale_floor: float = 1e-6, scale_warn: float = 1.0,
                 zero_grad_window: int = 200,
                 zero_grad_abort_frac: float = 0.5,
                 zero_grad_min_obs: int = 50,
                 logger=None) -> None:
        self.dt_bound = dt_bound
        self.interval = interval
        self.delta_warn = delta_warn_mult * dt_bound
        self.delta_abort = delta_abort_mult * dt_bound
        self.ratio_warn = ratio_warn
        # TWO-SIDED. Every original rule was built around job 558108, where
        # the TEACHER ran hot and crossed first. Job 559640 was the mirror
        # image: student 20x LARGER than teacher (ratio 0.051), which a
        # one-sided ">4.0" test cannot see.
        self.ratio_warn_low = ratio_warn_low
        self.teacher_abort = teacher_abort_frac * FP16_MAX
        # the student had NO absolute bound at all; 559640 reached
        # student_absmax 54 and climbing with nothing watching.
        self.student_abort = student_abort_frac * FP16_MAX
        # LOSS-TREND breaker. The skip-rate breaker only catches NON-FINITE
        # losses; 559640 stayed finite the whole way (0.11% non-finite) and
        # still diverged from 3.29 to >100. Guarded by loss_min_steps so
        # early-training noise cannot trip it.
        self.loss_window = loss_window
        self.loss_blowup_mult = loss_blowup_mult
        self.loss_min_steps = loss_min_steps
        self._loss_vals = deque(maxlen=loss_window)
        self._loss_min = None
        self.skip_window = skip_window
        self.skip_abort_frac = skip_abort_frac
        # ── MODE C: GRADIENT DEATH (job 567755 post-mortem) ──────────────
        # 567755 would have been reported as a clean 2,000-step survival. It
        # took 1,600 optimiser steps on EXACTLY ZERO gradients: 41 non-finite
        # gradient events each halved the GradScaler, driving it 65536 ->
        # 2.98e-08 (2^-41), at which point the fp16 gradients underflow to
        # zero. Every existing rule stayed silent -- the forward stayed
        # FINITE (skip-rate breaker quiet) and the loss FELL (trend breaker
        # quiet). The loss fell because Adam's weight_decay=1e-4 keeps
        # shrinking weights with no data gradient at all, collapsing the
        # model toward a trivial all-background predictor. That is collapse,
        # not learning, and nothing was reading the two observables that
        # showed it.
        #
        # SCALE FLOOR, justified against measurement (12 runs on disk):
        #   healthy minima 128 / 128 / 256 / 512 / 512 / 512 (6 clean runs)
        #   collapses      2.98e-08, 4.2e-22, 3.7e-09, 5.8e-11, 1.5e-36
        # Principled line: the scaler exists to LIFT gradients into fp16
        # range, so scale < 1.0 means it is now ATTENUATING them -- the
        # opposite of its purpose, and never a healthy state. That is the
        # WARN level (128x below the lowest healthy observation, and it
        # separates all 6 clean from all 6 failing runs perfectly).
        # ABORT is set two orders lower still, at 1e-6, for ONE reason:
        # scale<1.0 fires on job 561546 at step 6968, but 561546 aborted
        # naturally at 7667 and job 567949 REPLAYS that trajectory to test
        # the sec 7.14 slow-mode rules. An abort at 6968 would truncate 700
        # steps of the approach window and destroy the experiment. At 1e-6
        # it fires at 7656 -- 11 steps before the natural abort -- so the
        # replay is preserved. Do not raise this floor while 567949 is
        # pending without re-checking that interaction.
        self.scale_floor = float(scale_floor)
        self.scale_warn = float(scale_warn)
        self._scale_warned = False
        # ZERO-GRAD RULE: the direct mode-C detector -- "the optimiser is
        # stepping on nothing". Requires stepped=1, so a SKIPPED step (which
        # correctly zeroes grads) can never trigger it. Verified over all 12
        # runs on disk: fires ONLY on 567755 (step 347) and on nothing else,
        # including the five other failures and all six clean runs. Peak
        # trailing fraction elsewhere is 0.030, so 0.5 has a wide margin.
        self.zero_grad_window = zero_grad_window
        self.zero_grad_abort_frac = zero_grad_abort_frac
        self.zero_grad_min_obs = zero_grad_min_obs
        self._zero_grad = deque(maxlen=zero_grad_window)
        self._grad_finite = deque(maxlen=zero_grad_window)
        self._grad_norms = deque(maxlen=zero_grad_window)
        self._scale = float('nan')
        self.log = logger or print
        self._loss_finite = deque(maxlen=skip_window)
        self.history = []

    # -- called every step (cheap) ---------------------------------------
    def observe_step(self, loss_is_finite: bool, loss_value=None,
                     scale=None, grad_norm=None, stepped=None) -> None:
        self._loss_finite.append(bool(loss_is_finite))
        if loss_value is not None and loss_value == loss_value:
            self._loss_vals.append(float(loss_value))
            if len(self._loss_vals) == self.loss_window:
                m = sum(self._loss_vals) / len(self._loss_vals)
                if self._loss_min is None or m < self._loss_min:
                    self._loss_min = m
        # mode-C observables. All three optional so older call sites keep
        # working; when absent the mode-C rules simply never have data.
        if scale is not None:
            self._scale = float(scale)
        if grad_norm is not None:
            g = float(grad_norm)
            finite = (g == g)                      # NaN-safe
            self._grad_finite.append(finite)
            if finite:
                self._grad_norms.append(g)
                # ZERO only counts when the optimiser ACTUALLY STEPPED: a
                # skipped step legitimately has no gradient to speak of.
                self._zero_grad.append(bool(g == 0.0 and stepped))
            else:
                self._zero_grad.append(False)

    # -- instrument panel: the six numbers that make mode C obvious -------
    def panel(self) -> Dict:
        """Cheap per-step observables, logged on EVERY run (not just probe
        runs). 567755's failure was invisible for 1,600 steps purely because
        nothing recorded fraction_nonzero_grad or median_grad_norm."""
        gf = self._grad_finite
        zg = self._zero_grad
        gn = sorted(self._grad_norms)
        return {
            'scale': self._scale,
            'fraction_finite_grad': (sum(gf) / len(gf)) if gf else float('nan'),
            'fraction_nonzero_grad': (1.0 - sum(zg) / len(zg)) if zg
                                     else float('nan'),
            'median_grad_norm': gn[len(gn) // 2] if gn else float('nan'),
            'loss_trend': self.trend_ratio(),
        }

    def nonfinite_frac(self) -> float:
        if not self._loss_finite:
            return 0.0
        return 1.0 - sum(self._loss_finite) / len(self._loss_finite)

    def trend_ratio(self) -> float:
        """The loss-trend breaker's own statistic (trailing-window mean over
        running window-mean minimum), exposed read-only so the sec-7.14 grad
        probe can log it on the same per-step timeline it aborts on. NaN
        until the first full window."""
        if self._loss_min is None or len(self._loss_vals) < self.loss_window:
            return float("nan")
        return (sum(self._loss_vals) / len(self._loss_vals)) / self._loss_min

    # -- called every step; does real work every `interval` --------------
    def check(self, step: int, model) -> Optional[Dict]:
        # circuit breaker runs EVERY step: it is the one that bounds wasted
        # compute, and a full window is exactly what we do not want to wait for
        if len(self._loss_finite) == self.skip_window:
            frac = self.nonfinite_frac()
            if frac > self.skip_abort_frac:
                raise GateAbort(
                    'step %d: %.0f%% of the last %d steps had a NON-FINITE '
                    'loss (limit %.0f%%). This is a true forward failure, not '
                    'GradScaler backoff (backoff keeps the loss finite).'
                    % (step, 100 * frac, self.skip_window,
                       100 * self.skip_abort_frac))

        # LOSS-TREND breaker, every step alongside the skip-rate one.
        if (self._loss_min is not None and step >= self.loss_min_steps
                and len(self._loss_vals) == self.loss_window):
            mean = sum(self._loss_vals) / len(self._loss_vals)
            if mean > self.loss_blowup_mult * self._loss_min:
                raise GateAbort(
                    'step %d: trailing-%d mean loss %.3f exceeds %gx its own '
                    'running minimum %.3f. The loss is FINITE and diverging '
                    '-- the skip-rate breaker cannot see this (job 559640 ran '
                    '0.11%% non-finite while going from 3.29 to >100).'
                    % (step, self.loss_window, mean, self.loss_blowup_mult,
                       self._loss_min))

        # ── MODE C, both rules, every step alongside the other breakers ──
        if self._scale == self._scale and self._scale < self.scale_floor:
            raise GateAbort(
                'step %d: GradScaler scale %.3g fell below the floor %.3g. '
                'The scaler exists to LIFT gradients into fp16 range; below '
                '1.0 it attenuates them, and this far down the scaled '
                'gradients UNDERFLOW TO ZERO -- the optimiser then steps on '
                'nothing while the loss stays finite and even falls (Adam '
                'weight decay shrinking weights toward a trivial '
                'predictor). Healthy runs hold 128-512 (6 runs measured). '
                'This is job 567755\'s silent failure.'
                % (step, self._scale, self.scale_floor))
        if (self._scale == self._scale and self._scale < self.scale_warn
                and not self._scale_warned):
            self._scale_warned = True
            self.log('[gate] WARN step %d: GradScaler scale %.3g < %.3g -- '
                     'the scaler is now ATTENUATING gradients. Every healthy '
                     'run measured holds 128-512; every collapsed run passes '
                     'through here. Not yet an abort (floor %.3g).'
                     % (step, self._scale, self.scale_warn, self.scale_floor))
        if len(self._zero_grad) >= self.zero_grad_min_obs:
            zfrac = sum(self._zero_grad) / len(self._zero_grad)
            if zfrac > self.zero_grad_abort_frac:
                raise GateAbort(
                    'step %d: %.0f%% of the last %d STEPPED updates had '
                    'grad_norm EXACTLY 0.0 (limit %.0f%%). The optimiser is '
                    'stepping on nothing: this is gradient death, not '
                    'convergence. Loss may still be FALLING -- Adam weight '
                    'decay alone shrinks the weights -- so neither the '
                    'skip-rate nor the loss-trend breaker can see it. '
                    'Measured peak elsewhere across 11 other runs: 0.030.'
                    % (step, 100 * zfrac, len(self._zero_grad),
                       100 * self.zero_grad_abort_frac))

        if step % self.interval:
            return None

        rec = {'step': step, 'nonfinite_frac': round(self.nonfinite_frac(), 4)}
        scan = getattr(getattr(getattr(model, 'lc', None), 'cssm', None),
                       'scan', None)
        ds = getattr(scan, 'last_delta_stats', None) if scan else None
        if ds:
            rec.update(ds)
            if ds['delta_max'] > self.delta_abort:
                raise GateAbort(
                    'step %d: delta_max %.4f > %.4f (%gx dt_bound=%.2f). The '
                    'soft bound cannot be exceeded this far unless it is '
                    'missing or bypassed.'
                    % (step, ds['delta_max'], self.delta_abort,
                       self.delta_abort / self.dt_bound, self.dt_bound))
            if ds['delta_max'] > self.delta_warn:
                self.log('[gate] WARN step %d: delta_max %.4f > %.4f '
                         '(2x dt_bound); sat_frac %.4f'
                         % (step, ds['delta_max'], self.delta_warn,
                            ds.get('sat_frac', float('nan'))))

        ts = getattr(model, 'last_teacher_stats', None)
        if ts:
            rec.update(ts)
            if ts['teacher_absmax'] > self.teacher_abort:
                raise GateAbort(
                    'step %d: teacher |f_out|max %.1f > %.1f (0.25 x fp16 '
                    'ceiling). The teacher is fed the DENSE collaborator sum '
                    'and overflows first; this is the 558108 signature.'
                    % (step, ts['teacher_absmax'], self.teacher_abort))
            if ts.get('student_absmax', 0.0) > self.student_abort:
                raise GateAbort(
                    'step %d: student |f_out|max %.1f > %.1f (0.25 x fp16 '
                    'ceiling). Mirror of the teacher bound, which existed '
                    'alone until job 559640 diverged on the student side.'
                    % (step, ts['student_absmax'], self.student_abort))
            if ts['ratio'] > self.ratio_warn:
                self.log('[gate] WARN step %d: teacher/student magnitude '
                         'ratio %.2f > %.1f (teacher |max| %.1f, student '
                         '%.1f)' % (step, ts['ratio'], self.ratio_warn,
                                    ts['teacher_absmax'],
                                    ts['student_absmax']))
            elif ts['ratio'] < self.ratio_warn_low:
                self.log('[gate] WARN step %d: teacher/student magnitude '
                         'ratio %.4f < %.2f -- the STUDENT is %.1fx the '
                         'teacher (teacher |max| %.1f, student %.1f). This '
                         'is the 559640 direction.'
                         % (step, ts['ratio'], self.ratio_warn_low,
                            (1.0 / ts['ratio']) if ts['ratio'] else float('inf'),
                            ts['teacher_absmax'], ts['student_absmax']))

        self.history.append(rec)
        return rec
