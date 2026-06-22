# Griffin Dataset Visualisation

Tools for exploring and visualising the Griffin aerial-ground cooperative
perception dataset (arXiv:2503.06983, 2025), plus fault-injection and
information-quality utilities for robustness research.

## Repository structure

```
griffin-visualisation/
  src/
    download_griffin.py        download a subset from Hugging Face (CLI + importable)
    data_loaders.py            file I/O: images, LiDAR, poses, calibration, labels
    transforms.py              projection and coordinate transforms (ego frame)
    visualisation.py           plotting helpers
    fault_injectors/           fault injection package
      missing_modality.py      Failure Mode 1: Bernoulli sensor dropout
      temporal_misalignment.py Failure Mode 2: stale-image index shifting
    info_quality/              information-quality (mutual information) package
      estimators.py            InfoNCE + SMILE MI lower-bound estimators (agnostic core)
      preprocessing.py         seeding, standardisation, PCA
      feature_extraction.py    generic hook-based feature collection (only model-aware file)
      reporting.py             tables, fusion summary, comparison plots
      run_mi.py                CLI: python -m src.info_quality.run_mi
      tests/                   pytest suite (synthetic ground-truth)
    __init__.py
  examples/
    collect_bevfusion_features.py   BEVFusion -> FeatureCollector adapter
  notebooks/
    quick_start.ipynb                      load a frame and visualise it
    griffin_visualisation_tutorial.ipynb   full walkthrough
    temporal_animation.ipynb               multi-camera animation
    viser_viewer.py                        interactive 3D viewer (world frame)
    fault_injection_visualisation.ipynb    clean vs faulty, incl. compare animations
  docs/
    tutorial_guide.md
    coordinate_transformation.md           coordinate maths from scratch
    information_quality.md                  mutual information for fusion robustness
    animation_deep_dive.md
  requirements.txt
  requirements-info-quality.txt            extra deps for the info_quality module
```

## Key fact about coordinates

Griffin LiDAR points and 3D annotations are in the **ego frame** (the car is at
the origin). Projecting them onto a camera is ego -> sensor -> image directly,
with no vehicle-pose transform. See `docs/coordinate_transformation.md`.

## Quick start

```bash
pip install -r requirements.txt

# 1. Download a subset (interactive, or use flags)
python src/download_griffin.py --list
python src/download_griffin.py --subset griffin_50scenes_25m --minimal

# 2. Explore
jupyter notebook notebooks/quick_start.ipynb
```

### Downloading the data

`src/download_griffin.py` works both as a CLI and as an importable module.

```bash
python src/download_griffin.py                 # fully interactive
python src/download_griffin.py --list          # print the catalogue
python src/download_griffin.py --subset griffin_50scenes_25m --minimal
python src/download_griffin.py --subset griffin_50scenes_25m --all
```

```python
from src.download_griffin import download_subset, list_catalogue
download_subset('griffin_50scenes_25m', files='minimal', dest='./datasets')
```

The `--minimal` set (metadata + LiDAR + front + drone-bottom cameras, ~46 GB) is
enough for all the visualisation tools.

## Information quality (mutual information)

`src/info_quality` measures how much task-relevant information a learned
representation carries about the target, using mutual information lower bounds.
For fusion it answers: does the fused representation carry more about the target
than the best single modality, and how does that margin degrade under fault
injection?

The core quantity is the fusion gain

```
delta_I = I(Z_fused; Y) - max(I(Z_camera; Y), I(Z_lidar; Y))
```

Positive `delta_I` means fusion is synergistic. Two estimators (InfoNCE and
SMILE) are run so conclusions do not hinge on a single estimator. MI is always
reported as a lower bound, never a point estimate.

```bash
pip install -r requirements-info-quality.txt

# From saved feature arrays (an .npz or a pickled dict of {name: (N, d)} + Y):
python -m src.info_quality.run_mi --input features.npz --plot
```

```python
from src.info_quality.estimators import SMILEEstimator, delta_information

mi = {name: SMILEEstimator(holdout=0.3).estimate(Z, Y).mi_nats
      for name, Z in representations.items()}
gain = delta_information(mi, fused_key='Z_fused',
                         unimodal_keys=['Z_camera', 'Z_lidar'])
```

To pull features out of any PyTorch model, name the modules to tap and let
`FeatureCollector` hook them; `examples/collect_bevfusion_features.py` is a
worked BEVFusion adapter. See `docs/information_quality.md` for the full
explainer, the in-sample-bias caveat, and the fault-injection integration loop.

## Links

- Paper: https://arxiv.org/abs/2503.06983 (Griffin, arXiv:2503.06983, 2025)
- Dataset: https://huggingface.co/datasets/wjh-svm/Griffin
- Code: https://github.com/wang-jh18-SVM/Griffin
