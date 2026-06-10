"""
visualisation.py
----------------
Plotting utilities for the Griffin dataset.

LiDAR points and annotations are in the EGO frame (car at origin), so BEV and
front-view plots use the raw coordinates directly — no ego-position subtraction.

All functions return the matplotlib figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .transforms import project_ego_to_img, ego_box_corners_3d, ann_to_ego_corners_bev

CAT_COLORS = {
    'car':        '#2196F3',
    'pedestrian': '#FF5722',
    'truck':      '#9C27B0',
    'bus':        '#FF9800',
    'motorcycle': '#4CAF50',
    'bicycle':    '#00BCD4',
}


# ── Camera views ───────────────────────────────────────────────────────────

def plot_surround_cameras(frames_dict, title='Surround view', figsize=(14, 8)):
    """Display up to four camera images in a 2x2 grid."""
    order = ['front', 'right', 'back', 'left']
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13)
    for ax, cam in zip(axes.flat, order):
        if cam in frames_dict:
            ax.imshow(frames_dict[cam])
        ax.set_title(f'CAM_{cam.upper()}', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    return fig


# ── LiDAR BEV (ego frame: car at origin) ───────────────────────────────────

def plot_bev(pts, figsize=(14, 6), xlim=(-60, 60), ylim=(-60, 60)):
    """
    Bird's-eye view of an ego-frame LiDAR cloud. Car is at the origin.

    Parameters
    ----------
    pts        : np.ndarray (N, 4)  [x, y, z, intensity] in ego frame.
    figsize    : tuple
    xlim, ylim : tuple              display range (metres).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='plasma', s=0.5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='height z (m)', shrink=0.8)
    ax.set_xlabel('x — forward (m)'); ax.set_ylabel('y — left (m)')
    ax.set_title('BEV — coloured by height')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')

    ax = axes[1]
    vmax_i = pts[:, 3].mean() + 2 * pts[:, 3].std()
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap='viridis',
                    s=0.5, alpha=0.7, vmin=0, vmax=vmax_i)
    plt.colorbar(sc, ax=ax, label='intensity', shrink=0.8)
    ax.set_xlabel('x — forward (m)'); ax.set_ylabel('y — left (m)')
    ax.set_title('BEV — coloured by intensity')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')

    # Mark the ego/vehicle origin
    for ax in axes:
        ax.plot(0, 0, 'w^', markersize=8, markeredgecolor='black')

    plt.suptitle("LiDAR bird's-eye view (ego frame, car at origin)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_bev_with_boxes(pts, anns, figsize=(9, 9), xlim=(-60, 60), ylim=(-60, 60)):
    """BEV with ego-frame LiDAR points and 3D box footprints."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='gray', s=0.3, alpha=0.4)

    for ann in anns:
        cat    = ann.get('category', 'car').lower()
        color  = CAT_COLORS.get(cat, '#ffffff')
        poly   = plt.Polygon(ann_to_ego_corners_bev(ann), fill=False,
                             edgecolor=color, linewidth=1.5)
        ax.add_patch(poly)
        ax.text(ann['x'], ann['y'], cat[0].upper(),
                ha='center', va='center', fontsize=6, color=color, weight='bold')

    ax.plot(0, 0, 'w^', markersize=10, markeredgecolor='black', zorder=10)
    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('x — forward (m)'); ax.set_ylabel('y — left (m)')
    ax.set_title('BEV with 3D bounding boxes (ego frame)')
    ax.set_facecolor('#1a1a2e'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ── Front view (ego frame) ─────────────────────────────────────────────────

def plot_front_view(pts, figsize=(14, 6), az_range=(-45, 45), el_range=(-15, 15)):
    """
    LiDAR front view: azimuth vs elevation, from ego-frame points.

    Parameters
    ----------
    pts      : np.ndarray (N, 4)  ego-frame points.
    az_range : tuple              azimuth FoV (degrees).
    el_range : tuple              elevation FoV (degrees).
    """
    front   = pts[pts[:, 0] > 0]
    dist_xy = np.sqrt(front[:, 0]**2 + front[:, 1]**2)
    azimuth   = np.degrees(np.arctan2(front[:, 1], front[:, 0]))
    elevation = np.degrees(np.arctan2(front[:, 2], dist_xy))

    fov = ((azimuth   >= az_range[0]) & (azimuth   <= az_range[1]) &
           (elevation >= el_range[0]) & (elevation <= el_range[1]))
    az, el = azimuth[fov], elevation[fov]

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    sc0 = axes[0].scatter(az, el, c=front[fov, 3], cmap='hot', s=1, alpha=0.9)
    plt.colorbar(sc0, ax=axes[0], label='intensity', shrink=0.8)
    axes[0].set_ylabel('elevation (deg)')
    axes[0].set_title('LiDAR front view — coloured by intensity')
    axes[0].set_facecolor('#0a0a0a')

    sc1 = axes[1].scatter(az, el, c=dist_xy[fov], cmap='jet_r', s=1, alpha=0.9)
    plt.colorbar(sc1, ax=axes[1], label='depth (m)', shrink=0.8)
    axes[1].set_xlabel('azimuth (deg)'); axes[1].set_ylabel('elevation (deg)')
    axes[1].set_title('LiDAR front view — coloured by depth')
    axes[1].set_facecolor('#0a0a0a')

    plt.tight_layout()
    return fig


# ── Sensor fusion overlay ──────────────────────────────────────────────────

def plot_fusion(img, uvd, camera='front', figsize=(16, 5), vmin=1, vmax=50):
    """Side-by-side RGB and RGB + projected LiDAR points."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(img)
    axes[0].set_title(f'RGB only — {camera} camera')
    axes[0].axis('off')

    axes[1].imshow(img)
    if len(uvd) > 0:
        sc = axes[1].scatter(uvd[:, 0], uvd[:, 1], c=uvd[:, 2],
                             cmap='jet_r', s=2, alpha=0.9, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=axes[1], label='depth (m)', shrink=0.85)
    else:
        axes[1].text(0.5, 0.5, 'No points projected',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=14, color='red')
    axes[1].set_title(f'RGB + LiDAR — {camera}  ({len(uvd):,} pts)')
    axes[1].axis('off')
    plt.suptitle('Sensor fusion — LiDAR projected onto RGB', fontsize=13)
    plt.tight_layout()
    return fig


# ── 3D boxes on image ──────────────────────────────────────────────────────

def _draw_box_edges(ax, c, color):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        if c[i, 2] > 0 and c[j, 2] > 0:
            ax.plot([c[i, 0], c[j, 0]], [c[i, 1], c[j, 1]],
                    color=color, linewidth=1.2, alpha=0.85)


def plot_boxes_on_image(img, anns, K, T_ego_to_sensor, camera='front', figsize=(14, 7)):
    """Draw projected 3D bounding boxes onto an RGB image."""
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    for ann in anns:
        cat    = ann.get('category', 'car').lower()
        color  = CAT_COLORS.get(cat, '#ffffff')
        corners_uvd = project_ego_to_img(ego_box_corners_3d(ann), K, T_ego_to_sensor, h, w)
        if (corners_uvd[:, 2] > 0).sum() >= 4:
            _draw_box_edges(ax, corners_uvd, color)
            visible = corners_uvd[corners_uvd[:, 2] > 0]
            if len(visible):
                top = visible[np.argmin(visible[:, 1])]
                if 0 <= top[0] < w and 0 <= top[1] < h:
                    ax.text(top[0], top[1] - 5, cat, fontsize=7, color=color, weight='bold')

    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.6)
    ax.set_title(f'3D bounding boxes — {camera} camera')
    ax.axis('off')
    plt.tight_layout()
    return fig