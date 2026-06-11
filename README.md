# Griffin Dataset Visualisation

Tools for exploring and visualising the Griffin aerial-ground cooperative
perception dataset (AAAI 2026), plus fault-injection utilities for robustness
research.

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
    __init__.py
  notebooks/
    quick_start.ipynb                      load a frame and visualise it
    griffin_visualisation_tutorial.ipynb   full walkthrough
    temporal_animation.ipynb               multi-camera animation
    viser_viewer.py                        interactive 3D viewer (world frame)
    fault_injection_visualisation.ipynb    clean vs faulty, incl. compare animations
  docs/
    tutorial_guide.md
    coordinate_transformation.md           coordinate maths from scratch
    animation_deep_dive.md
  requirements.txt
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

## Links

- Paper: https://arxiv.org/abs/2503.06983
- Dataset: https://huggingface.co/datasets/wjh-svm/Griffin
- Code: https://github.com/wang-jh18-SVM/Griffin
