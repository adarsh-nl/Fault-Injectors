"""
render_fault_animations.py
--------------------------
Render temporal animations comparing CLEAN vs FAULTY (Missing Modality) data,
for every camera on the ground vehicle and the UAV, and for the LiDAR.

For each sensor stream it writes three MP4s into
results/fault_injector_visualisation/:

    <sensor>_clean.mp4     the original stream
    <sensor>_faulty.mp4    after Bernoulli dropout (black frame / empty cloud)
    <sensor>_compare.mp4   clean and faulty side by side

Image cameras use RGB dropout (p_drop_rgb). The LiDAR (vehicle only) uses LiDAR
dropout (p_drop_lidar) and is shown as a bird's-eye view in the ego frame.

Run
---
    python render_fault_animations.py

Needs ffmpeg (apt-get install -y ffmpeg  or  conda install -c conda-forge ffmpeg).
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import load_lidar, get_file_lists
from src.fault_injectors import MissingModalityInjector, drop_image, drop_points


# ── Configuration ──────────────────────────────────────────────────────────

DATASET_ROOT = '../datasets/griffin_50scenes_25m/griffin-release/griffin_50scenes_25m/griffin-release'
VEH   = os.path.join(DATASET_ROOT, 'vehicle-side')
DRONE = os.path.join(DATASET_ROOT, 'drone-side')

OUTPUT_DIR = '../results/fault_injector_visualisation'

FRAME_START = 600          # a driving range within one scene
FRAME_END   = 670          # exclusive
FPS         = 10
DOWNSAMPLE  = 3            # image downsample factor (3 keeps files small)

P_DROP_RGB   = 0.35        # per-frame camera drop probability
P_DROP_LIDAR = 0.50        # per-frame LiDAR drop probability
SEED         = 0           # reproducible dropout schedule

# Which streams to render. Comment any out to skip.
VEH_CAMERAS   = ['front', 'back', 'left', 'right']
DRONE_CAMERAS = ['front', 'back', 'left', 'right', 'bottom']
RENDER_LIDAR  = True

SAVE_CLEAN   = True
SAVE_FAULTY  = True
SAVE_COMPARE = True

BEV_RANGE = 60             # metres each side for the LiDAR BEV


# ── Helpers ────────────────────────────────────────────────────────────────

def load_image_ds(path, ds=1):
    img = PILImage.open(path).convert('RGB')
    if ds > 1:
        w, h = img.size
        img = img.resize((w // ds, h // ds), PILImage.BILINEAR)
    return np.array(img)


def writer():
    return FFMpegWriter(fps=FPS, bitrate=3000, metadata={'title': 'Griffin fault injection'})


def _save(anim, path):
    anim.save(path, writer=writer(), dpi=100)
    plt.close(anim._fig)


# ── Image stream animations ────────────────────────────────────────────────

def render_image_single(images, path, title):
    """Single-panel animation of an image sequence."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#0d0d0d')
    ax.axis('off')
    im = ax.imshow(images[0], aspect='auto')
    txt = ax.set_title(title.format(i=FRAME_START), color='white', fontsize=10)

    def update(k):
        im.set_data(images[k])
        txt.set_text(title.format(i=FRAME_START + k))
        return [im, txt]

    anim = FuncAnimation(fig, update, frames=len(images), interval=1000 // FPS, blit=False)
    anim._fig = fig
    _save(anim, path)


def render_image_compare(clean, faulty, masks, path, cam_name):
    """Side-by-side clean | faulty image animation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), facecolor='#0d0d0d')
    for ax in axes:
        ax.axis('off')
    im0 = axes[0].imshow(clean[0],  aspect='auto')
    im1 = axes[1].imshow(faulty[0], aspect='auto')
    t0 = axes[0].set_title('CLEAN', color='white', fontsize=11)
    t1 = axes[1].set_title('FAULTY', color='white', fontsize=11)
    sup = fig.suptitle('', color='white', fontsize=12, fontfamily='monospace')

    def update(k):
        im0.set_data(clean[k]); im1.set_data(faulty[k])
        state = 'kept' if masks[k] else 'DROPPED (black)'
        t1.set_text(f'FAULTY  [{state}]')
        sup.set_text(f'{cam_name}   frame {FRAME_START + k}')
        return [im0, im1, t1, sup]

    anim = FuncAnimation(fig, update, frames=len(clean), interval=1000 // FPS, blit=False)
    anim._fig = fig
    _save(anim, path)


# ── LiDAR BEV animations ────────────────────────────────────────────────────

def _bev_ax(ax):
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(-BEV_RANGE, BEV_RANGE); ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect('equal'); ax.tick_params(colors='#555', labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.plot(0, 0, 'w^', markersize=8, zorder=10)   # ego marker


def render_bev_single(pts_list, path, title):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
    _bev_ax(ax)
    sc = ax.scatter([], [], c=[], cmap='plasma', s=1.2, vmin=-2, vmax=6)
    txt = ax.set_title(title.format(i=FRAME_START), color='white', fontsize=10)

    def update(k):
        p = pts_list[k]
        if len(p) > 0:
            sc.set_offsets(p[:, :2]); sc.set_array(p[:, 2])
        else:
            sc.set_offsets(np.empty((0, 2))); sc.set_array(np.array([]))
        txt.set_text(title.format(i=FRAME_START + k))
        return [sc, txt]

    anim = FuncAnimation(fig, update, frames=len(pts_list), interval=1000 // FPS, blit=False)
    anim._fig = fig
    _save(anim, path)


def render_bev_compare(clean_pts, faulty_pts, masks, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0d0d0d')
    for ax in axes:
        _bev_ax(ax)
    sc0 = axes[0].scatter([], [], c=[], cmap='plasma', s=1.2, vmin=-2, vmax=6)
    sc1 = axes[1].scatter([], [], c=[], cmap='plasma', s=1.2, vmin=-2, vmax=6)
    axes[0].set_title('CLEAN', color='white', fontsize=11)
    t1 = axes[1].set_title('FAULTY', color='white', fontsize=11)
    sup = fig.suptitle('', color='white', fontsize=12, fontfamily='monospace')

    def update(k):
        p = clean_pts[k]
        sc0.set_offsets(p[:, :2]); sc0.set_array(p[:, 2])
        f = faulty_pts[k]
        if len(f) > 0:
            sc1.set_offsets(f[:, :2]); sc1.set_array(f[:, 2])
        else:
            sc1.set_offsets(np.empty((0, 2))); sc1.set_array(np.array([]))
        state = 'kept' if masks[k] else 'DROPPED (empty)'
        t1.set_text(f'FAULTY  [{state}]')
        sup.set_text(f'LiDAR BEV   frame {FRAME_START + k}')
        return [sc0, sc1, t1, sup]

    anim = FuncAnimation(fig, update, frames=len(clean_pts), interval=1000 // FPS, blit=False)
    anim._fig = fig
    _save(anim, path)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = get_file_lists(VEH, DRONE)
    N = FRAME_END - FRAME_START
    saved = []

    # Precompute reproducible dropout schedules (one per modality)
    rgb_sched   = MissingModalityInjector(p_drop_rgb=P_DROP_RGB, seed=SEED).simulate_sequence(N)['m_rgb']
    lidar_sched = MissingModalityInjector(p_drop_lidar=P_DROP_LIDAR, seed=SEED).simulate_sequence(N)['m_lidar']

    print(f'Frames {FRAME_START}..{FRAME_END-1}  ({N} frames)')
    print(f'RGB drop p={P_DROP_RGB}: {int((rgb_sched==0).sum())}/{N} frames dropped')
    print(f'LiDAR drop p={P_DROP_LIDAR}: {int((lidar_sched==0).sum())}/{N} frames dropped')
    print(f'Output -> {OUTPUT_DIR}\n')

    # ── Camera streams ─────────────────────────────────────────────────────
    camera_jobs = (
        [('vehicle_' + c, VEH, c)   for c in VEH_CAMERAS] +
        [('drone_'   + c, DRONE, c) for c in DRONE_CAMERAS]
    )

    for name, root, sensor in camera_jobs:
        img_files = sorted(glob.glob(os.path.join(root, 'camera', sensor, '*.png')))
        if len(img_files) < FRAME_END:
            print(f'[skip] {name}: only {len(img_files)} images')
            continue
        print(f'[{name}] loading frames...')
        clean = [load_image_ds(img_files[FRAME_START + k], DOWNSAMPLE) for k in range(N)]
        faulty = [clean[k] if rgb_sched[k] else drop_image(clean[k]) for k in range(N)]

        if SAVE_CLEAN:
            p = os.path.join(OUTPUT_DIR, f'{name}_clean.mp4')
            render_image_single(clean, p, name + '  CLEAN  frame {i}'); saved.append(p)
        if SAVE_FAULTY:
            p = os.path.join(OUTPUT_DIR, f'{name}_faulty.mp4')
            render_image_single(faulty, p, name + '  FAULTY  frame {i}'); saved.append(p)
        if SAVE_COMPARE:
            p = os.path.join(OUTPUT_DIR, f'{name}_compare.mp4')
            render_image_compare(clean, faulty, rgb_sched, p, name); saved.append(p)
        print(f'[{name}] done.')

    # ── LiDAR stream (vehicle only) ────────────────────────────────────────
    if RENDER_LIDAR:
        print('[lidar_bev] loading point clouds...')
        clean_pts = [load_lidar(files['lidar_plys'][FRAME_START + k]) for k in range(N)]
        faulty_pts = [clean_pts[k] if lidar_sched[k] else drop_points(clean_pts[k]) for k in range(N)]

        if SAVE_CLEAN:
            p = os.path.join(OUTPUT_DIR, 'lidar_bev_clean.mp4')
            render_bev_single(clean_pts, p, 'LiDAR BEV  CLEAN  frame {i}'); saved.append(p)
        if SAVE_FAULTY:
            p = os.path.join(OUTPUT_DIR, 'lidar_bev_faulty.mp4')
            render_bev_single(faulty_pts, p, 'LiDAR BEV  FAULTY  frame {i}'); saved.append(p)
        if SAVE_COMPARE:
            p = os.path.join(OUTPUT_DIR, 'lidar_bev_compare.mp4')
            render_bev_compare(clean_pts, faulty_pts, lidar_sched, p); saved.append(p)
        print('[lidar_bev] done.')

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\nSaved {len(saved)} animations to {OUTPUT_DIR}:')
    for p in saved:
        print(f'  {os.path.basename(p)}')


if __name__ == '__main__':
    main()
