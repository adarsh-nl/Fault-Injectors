"""
render_fault_animations.py
--------------------------
Render temporal animations comparing CLEAN vs FAULTY (Missing Modality) data,
for every camera on the ground vehicle and the UAV, and for the LiDAR.

For each sensor stream it writes (up to) three clips into
results/fault_injector_visualisation/:

    <sensor>_clean      the original stream
    <sensor>_faulty     after Bernoulli dropout (black frame / empty cloud)
    <sensor>_compare    clean and faulty side by side

Image cameras use RGB dropout (p_drop_rgb). The LiDAR (vehicle only) uses LiDAR
dropout (p_drop_lidar) and is shown as a bird's-eye view in the ego frame.

Rendering scaffolding (image loading, the writer, BEV styling, the panel
animation loop) lives in ``src/anim_utils.py`` and is shared with the notebooks.

Run
---
    python -m src.render_fault_animations        # from the repo root

Writes MP4 if ffmpeg is available, otherwise falls back to GIF automatically.
"""

import os

import numpy as np

from src import load_lidar, get_file_lists
from src.fault_injectors import MissingModalityInjector, drop_image, drop_points
from src.anim_utils import (
    camera_specs, camera_image_files, load_image_ds,
    animate_image_panels, animate_bev_panels,
)


# ── Configuration ──────────────────────────────────────────────────────────
DATASET_ROOT = '../datasets/griffin_50scenes_25m/griffin-release/griffin_50scenes_25m/griffin-release'
VEH   = os.path.join(DATASET_ROOT, 'vehicle-side')
DRONE = os.path.join(DATASET_ROOT, 'drone-side')
OUTPUT_DIR = '../results/fault_injector_visualisation'

FRAME_START = 600          # a driving range within one scene
FRAME_END   = 670          # exclusive
FPS         = 10
DOWNSAMPLE  = 3            # image downsample factor (3 keeps files small)

P_DROP_RGB   = 0.35
P_DROP_LIDAR = 0.50
SEED         = 0

VEH_CAMERAS   = ['front', 'back', 'left', 'right']
DRONE_CAMERAS = ['front', 'back', 'left', 'right', 'bottom']
RENDER_LIDAR  = True

SAVE_CLEAN, SAVE_FAULTY, SAVE_COMPARE = True, True, True
BEV_RANGE = 60


def main():
    files = get_file_lists(VEH, DRONE)
    specs = camera_specs(VEH, DRONE)
    N = FRAME_END - FRAME_START
    saved = []

    # Reproducible dropout schedules (one per modality)
    rgb_sched   = MissingModalityInjector(p_drop_rgb=P_DROP_RGB, seed=SEED).simulate_sequence(N)['m_rgb']
    lidar_sched = MissingModalityInjector(p_drop_lidar=P_DROP_LIDAR, seed=SEED).simulate_sequence(N)['m_lidar']

    print(f'Frames {FRAME_START}..{FRAME_END - 1}  ({N} frames)')
    print(f'RGB   drop p={P_DROP_RGB}: {int((rgb_sched == 0).sum())}/{N} frames dropped')
    print(f'LiDAR drop p={P_DROP_LIDAR}: {int((lidar_sched == 0).sum())}/{N} frames dropped')
    print(f'Output -> {OUTPUT_DIR}\n')

    def frame_label(tag):
        return lambda k: f'{tag}  frame {FRAME_START + k}'

    def faulty_state(sched, dropped_word):
        return lambda k: f"FAULTY  [{'kept' if sched[k] else dropped_word}]"

    # ── Camera streams ─────────────────────────────────────────────────────
    for name in [f'vehicle_{c}' for c in VEH_CAMERAS] + [f'drone_{c}' for c in DRONE_CAMERAS]:
        root, sensor = specs[name]
        img_files = camera_image_files(root, sensor)
        if len(img_files) < FRAME_END:
            print(f'[skip] {name}: only {len(img_files)} images')
            continue
        print(f'[{name}] loading frames...')
        clean  = [load_image_ds(img_files[FRAME_START + k], DOWNSAMPLE) for k in range(N)]
        faulty = [clean[k] if rgb_sched[k] else drop_image(clean[k]) for k in range(N)]

        if SAVE_CLEAN:
            saved.append(animate_image_panels(
                [clean], N, f'{name}_clean', OUTPUT_DIR, FPS, titles=[frame_label(f'{name}  CLEAN')]))
        if SAVE_FAULTY:
            saved.append(animate_image_panels(
                [faulty], N, f'{name}_faulty', OUTPUT_DIR, FPS, titles=[frame_label(f'{name}  FAULTY')]))
        if SAVE_COMPARE:
            saved.append(animate_image_panels(
                [clean, faulty], N, f'{name}_compare', OUTPUT_DIR, FPS,
                titles=['CLEAN', faulty_state(rgb_sched, 'DROPPED (black)')],
                suptitle_fn=frame_label(name)))
        print(f'[{name}] done.')

    # ── LiDAR stream (vehicle only) ────────────────────────────────────────
    if RENDER_LIDAR:
        print('[lidar_bev] loading point clouds...')
        clean_pts  = [load_lidar(files['lidar_plys'][FRAME_START + k]) for k in range(N)]
        faulty_pts = [clean_pts[k] if lidar_sched[k] else drop_points(clean_pts[k]) for k in range(N)]

        if SAVE_CLEAN:
            saved.append(animate_bev_panels(
                [clean_pts], N, 'lidar_bev_clean', OUTPUT_DIR, FPS, BEV_RANGE,
                titles=[frame_label('LiDAR BEV  CLEAN')]))
        if SAVE_FAULTY:
            saved.append(animate_bev_panels(
                [faulty_pts], N, 'lidar_bev_faulty', OUTPUT_DIR, FPS, BEV_RANGE,
                titles=[frame_label('LiDAR BEV  FAULTY')]))
        if SAVE_COMPARE:
            saved.append(animate_bev_panels(
                [clean_pts, faulty_pts], N, 'lidar_bev_compare', OUTPUT_DIR, FPS, BEV_RANGE,
                titles=['CLEAN', faulty_state(lidar_sched, 'DROPPED (empty)')],
                suptitle_fn=frame_label('LiDAR BEV')))
        print('[lidar_bev] done.')

    print(f'\nSaved {len(saved)} animations to {OUTPUT_DIR}:')
    for p in saved:
        print(f'  {os.path.basename(p)}')


if __name__ == '__main__':
    main()
