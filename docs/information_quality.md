# Information Quality via Mutual Information

This module measures **information quality**: how much task-relevant information
a learned representation carries, and how that changes under fault injection. It
is the part of the research plan that answers "how do we measure information
quality?" for RQ2.

It is deliberately **architecture- and dataset-agnostic**. The estimators take
two plain arrays and return a number in nats. Nothing in the core knows about
BEVFusion, nuScenes, Griffin, cameras or LiDAR. The same code measures a
BEVFusion BEV feature on nuScenes, a Griffin drone-side voxel feature, or a
synthetic Gaussian with a known answer.

---

## 1. What is being measured

Information quality is operationalised as the **mutual information** between a
representation `Z` and a task-relevant target `Y`:

```
I(Z; Y) = how much knowing Z tells you about Y, in nats.
```

`I = 0` means `Z` carries nothing about the task. Larger `I` means more
task-relevant information. Two derived quantities matter for the research:

- **Fusion gain.** For a fused representation against its single modalities,

  ```
  delta_I = I(Z_fused; Y) - max( I(Z_camera; Y), I(Z_lidar; Y) )
  ```

  `delta_I > 0` means fusion is **synergistic** (the fused feature knows more
  than the best single sensor). `delta_I <= 0` means fusion is **redundant or
  lossy**.

- **Information retained under a fault.** Estimate `I(Z; Y)` for the same
  representation on clean inputs and on faulty inputs (across a severity sweep).
  The ratio or difference is how much task information survives the fault. This
  is the information-quality axis that complements the detection-accuracy axis
  in the results table of the plan.

---

## 2. Why two estimators

`I(Z; Y)` has no closed form for continuous neural features, so it must be
**estimated**, and every estimator is biased. Reporting two estimators with
different biases upgrades a claim from "fusion adds information" to "fusion adds
information under both InfoNCE and SMILE", which is much harder for a reviewer to
attribute to an estimator artefact.

| | InfoNCE | SMILE |
|---|---|---|
| Family | Contrastive lower bound | Variational lower bound (clipped) |
| Bound | `I >= log(K) - L_NCE` | clipped Donsker–Varadhan |
| Ceiling | Capped at `log(K)` (with `K = N`) | None |
| Variance | Low, very stable | Higher, controlled by the clip |
| Fails when | `N` small (bound saturates at `log N`) | clip too large (variance) |

**InfoNCE** trains a critic to match `z_i` to its partner `y_i` against
in-batch negatives; the cross-entropy loss yields the bound. It is rock-solid
but cannot report more than `log(N)` nats, so with `N = 81` it tops out at
`log(81) ≈ 4.39` and can fail to separate representations that are all
informative.

**SMILE** (Song & Ermon, 2020) is a variance-reduced repair of MINE. MINE uses a
*detached* EMA denominator, so the statistics network can drive its score to
infinity with no gradient penalty — the estimate collapses (a "death spiral").
SMILE clips the log density ratio to `[-clip, +clip]` inside a **fully
differentiable** denominator, which bounds the gradient and removes the
divergence. It has no `log(N)` ceiling, so it can separate representations that
InfoNCE cannot.

If the two estimators agree on the sign of `delta_I`, the conclusion is
estimator-agnostic. If they disagree at small `N`, the usual cause is the
InfoNCE ceiling — read the SMILE number and increase `N`.

---

## 3. The bias caveat (read before quoting absolute numbers)

Both bounds are trained on data, then read off. If you read them off **on the
same samples you trained on** (in-sample), the critic can memorise the pairing
and the bound is **optimistic**. The effect is real: in-sample InfoNCE reports
roughly 0.5 nats on genuinely *independent* data where the true MI is 0.

Two consequences:

1. **For relative comparisons under a fixed protocol** (clean vs faulty, modality
   vs fused), the bias is largely shared and cancels in the difference. In-sample
   estimation (`holdout=0`) is acceptable and is the default, which also matters
   in the small-`N` regime where you cannot spare a split.

2. **For absolute claims**, pass `holdout > 0`. The critic is trained on the
   training split and the bound is evaluated on held-out pairs, which removes the
   memorisation bias (held-out InfoNCE returns ~0 on independent data). Use this
   whenever `N` is large enough to spare a validation split.

Report MI values as **lower bounds**, never as exact point estimates.

---

## 4. Module layout

```
src/info_quality/
  estimators.py        InfoNCEEstimator, SMILEEstimator  (the agnostic core)
  preprocessing.py     standardise / PCA / seeding        (shared)
  feature_extraction.py FeatureCollector                  (the ONLY model-aware step)
  reporting.py         tables and plots
  run_mi.py            CLI: features file -> MI numbers
  tests/               closed-form validation against Gaussian ground truth
```

Everything except `feature_extraction.py` is independent of architecture and
dataset.

---

## 5. Usage

### 5.1 Estimating MI from arrays you already have

```python
import numpy as np
from src.info_quality import InfoNCEEstimator, SMILEEstimator, delta_information

# Z: (N, d_z) features, Y: (N, d_y) targets, rows aligned.
nce = InfoNCEEstimator(epochs=100, temperature=0.07, seed=0)
smi = SMILEEstimator(epochs=500, clip=5.0, seed=0)

mi = {
    'camera': nce.estimate(Z_camera, Y, pca_dims=32).mi_nats,
    'lidar':  nce.estimate(Z_lidar,  Y, pca_dims=32).mi_nats,
    'fused':  nce.estimate(Z_fused,  Y, pca_dims=32).mi_nats,
}
print('delta_I =', delta_information(mi, 'fused', ['camera', 'lidar']))
```

### 5.2 Collecting features from any model

`FeatureCollector` attaches forward hooks to modules you name and pools their
outputs to one row per sample. You supply the forward call and the label
extraction, because those differ across frameworks.

```python
from src.info_quality import FeatureCollector, FeatureSet

collector = FeatureCollector(model, taps={
    'camera': model.encoders['camera'].neck,     # any module reference
    'lidar':  model.encoders['lidar'].backbone,
    'fused':  model.fuser,
})
fs = collector.collect(
    loader,
    forward_fn=lambda m, batch: m(return_loss=False, **batch),
    label_fn=label_fn,           # batch -> (d_y,) target row
    n_samples=200,
)
collector.remove()
fs.save('features.npz')          # one array per tap, plus Y
```

To use a **different architecture**, change the `taps` dictionary and the two
callables. Nothing else changes. A worked BEVFusion adapter (the exact hook
points from the original notebook, expressed in this API) is in
`examples/collect_bevfusion_features.py`.

### 5.3 From the command line

```bash
pip install -r requirements-info-quality.txt

# Run both estimators on every representation in the file, plot, and report
# the fusion gain of Z_fused over the two modalities.
python -m src.info_quality.run_mi features.npz \
    --estimators infonce smile --pca-dims 32 \
    --fused Z_fused --unimodal Z_camera Z_lidar \
    --plot mi.png
```

The features file may be the `.npz` written by `FeatureSet.save`, or a pickled
dict such as the original notebook's `{Z_camera, Z_lidar, Z_fused, Y}`. Every 2D
array except the target key is treated as a representation.

---

## 6. How this plugs into fault-injection experiments

The natural loop for RQ2 is: for each fault injector and each severity level,
collect features on the corrupted inputs, estimate `I(Z; Y)`, and record it
alongside detection accuracy.

```python
for severity in [0.0, 0.25, 0.50, 0.75, 1.00]:
    faulty_loader = make_loader(injector, severity)
    fs = collector.collect(faulty_loader, forward_fn, label_fn)
    mi_fused = smi.estimate(fs.features['fused'], fs.Y, pca_dims=32).mi_nats
    record(severity, mi_fused)        # information-quality curve vs severity
```

A method whose `I(Z_fused; Y)` decays more slowly than the baselines' as
severity rises is more robust in the information sense, independent of any
particular detection head. That is the information-quality column of the results
table in the research plan, and it is a claim about the representation rather
than the downstream detector, which makes it architecture-portable.

---

## 7. Validation

The estimators are checked against a closed-form ground truth. For `Z, Y`
jointly Gaussian with per-dimension correlation `rho`,

```
I(Z; Y) = -(dim / 2) * log(1 - rho^2)   nats,
```

which `correlated_gaussians()` generates. The test suite confirms InfoNCE tracks
the signal and stays a lower bound, SMILE recovers the right magnitude, the
holdout split removes the in-sample bias on independent data, and runs are
reproducible under a fixed seed:

```bash
pytest src/info_quality/tests/test_estimators.py -v
```

---

## 8. References

- van den Oord, Li, Vinyals (2018). *Representation Learning with Contrastive
  Predictive Coding* (InfoNCE).
- Belghazi et al. (2018). *Mutual Information Neural Estimation* (MINE).
- Song, Ermon (2020). *Understanding the Limitations of Variational Mutual
  Information Estimators* (SMILE).
- Poole et al. (2019). *On Variational Bounds of Mutual Information*.
