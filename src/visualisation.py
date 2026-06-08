"""
visualisation.py
----------------
Plotting utilities for the Griffin dataset.

All functions return the matplotlib figure so callers can save or display it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .transforms import project_ego_to_img, ego_box_corners_3d, ann_to_ego_corners_bev

# ── Category colours ───────────────────────────────────────────────────────

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
    """
    Display up to four camera images in a 2x2 grid.

    Parameters
    ----------
    frames_dict : dict  Keys are camera names ('front','back','left','right'),
                        values are (H,W,3) uint8 numpy arrays.
    title       : str
    figsize     : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
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


# ── LiDAR BEV ─────────────────────────────────────────────────────────────

def plot_bev(pts, ego_x, ego_y, figsize=(14, 6), xlim=(-80, 80), ylim=(-80, 80)):
    """
    Bird's-eye view of a LiDAR point cloud, ego-centred.

    Parameters
    ----------
    pts          : np.ndarray (N, 4)  LiDAR points [x, y, z, intensity] in ENU.
    ego_x, ego_y : float              Ego vehicle world position for centring.
    figsize      : tuple
    xlim, ylim   : tuple              Display range in metres.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    dx = pts[:, 0] - ego_x
    dy = pts[:, 1] - ego_y

    ax = axes[0]
    sc = ax.scatter(dx, dy, c=pts[:, 2], cmap='plasma', s=0.5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='height z (m)', shrink=0.8)
    ax.set_xlabel('x relative to ego (m)')
    ax.set_ylabel('y relative to ego (m)')
    ax.set_title('BEV — coloured by height')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    ax = axes[1]
    vmax_i = pts[:, 3].mean() + 2 * pts[:, 3].std()
    sc = ax.scatter(dx, dy, c=pts[:, 3], cmap='viridis', s=0.5, alpha=0.7,
                    vmin=0, vmax=vmax_i)
    plt.colorbar(sc, ax=ax, label='intensity', shrink=0.8)
    ax.set_xlabel('x relative to ego (m)')
    ax.set_ylabel('y relative to ego (m)')
    ax.set_title('BEV — coloured by intensity')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    plt.suptitle("LiDAR bird's-eye view (ego-centred)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_bev_with_boxes(pts, anns, ego_x, ego_y,
                        figsize=(9, 9), xlim=(-60, 60), ylim=(-60, 60)):
    """
    BEV with LiDAR points and 3D bounding box footprints overlaid.

    Parameters
    ----------
    pts          : np.ndarray (N, 4)  LiDAR points in ENU frame.
    anns         : list of dict       Annotation list from load_labels_for_frame.
    ego_x, ego_y : float              Ego position for centring.
    figsize      : tuple
    xlim, ylim   : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pts[:, 0] - ego_x, pts[:, 1] - ego_y,
               c=pts[:, 2], cmap='gray', s=0.3, alpha=0.4)

    for ann in anns:
        cat    = ann.get('category', 'car').lower()
        color  = CAT_COLORS.get(cat, '#ffffff')
        corners = ann_to_ego_corners_bev(ann)
        poly   = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(poly)
        ax.text(ann['x'], ann['y'], cat[0].upper(),
                ha='center', va='center', fontsize=6, color=color, weight='bold')

    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel('x — forward (m)'); ax.set_ylabel('y — left (m)')
    ax.set_title('BEV with 3D bounding boxes')
    ax.set_facecolor('#1a1a2e'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ── Front view ─────────────────────────────────────────────────────────────

def plot_front_view(pts, ego_x, ego_y, figsize=(14, 6),
                    az_range=(-45, 45), el_range=(-15, 15)):
    """
    LiDAR front view: azimuth vs elevation, coloured by intensity and depth.

    Parameters
    ----------
    pts          : np.ndarray (N, 4)  LiDAR points in ENU frame.
    ego_x, ego_y : float              Ego position for centring.
    figsize      : tuple
    az_range     : tuple              Azimuth field of view (degrees).
    el_range     : tuple              Elevation field of view (degrees).

    Returns
    -------
    matplotlib.figure.Figure
    """
    pts_rel = pts.copy()
    pts_rel[:, 0] -= ego_x
    pts_rel[:, 1] -= ego_y

    front   = pts_rel[pts_rel[:, 0] > 0]
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
    axes[1].set_xlabel('azimuth (deg)')
    axes[1].set_ylabel('elevation (deg)')
    axes[1].set_title('LiDAR front view — coloured by depth')
    axes[1].set_facecolor('#0a0a0a')

    plt.tight_layout()
    return fig


# ── Sensor fusion overlay ──────────────────────────────────────────────────

def plot_fusion(img, uvd, camera='front', figsize=(16, 5),
                vmin=1, vmax=50):
    """
    Side-by-side: RGB image and RGB + projected LiDAR points.

    Parameters
    ----------
    img    : np.ndarray (H, W, 3)  RGB image.
    uvd    : np.ndarray (M, 3)     Projected points [u, v, depth].
    camera : str                    Camera name for title.
    figsize: tuple
    vmin, vmax : float              Depth colour scale range.

    Returns
    -------
    matplotlib.figure.Figure
    """
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


# ── 3D bounding boxes on image ─────────────────────────────────────────────

def _draw_box_edges(ax, corners_uvd, color):
    """Draw 12 edges of a 3D box. Internal helper."""
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        if corners_uvd[i, 2] > 0 and corners_uvd[j, 2] > 0:
            ax.plot([corners_uvd[i, 0], corners_uvd[j, 0]],
                    [corners_uvd[i, 1], corners_uvd[j, 1]],
                    color=color, linewidth=1.2, alpha=0.85)


def plot_boxes_on_image(img, anns, K, T_ego_to_sensor,
                        camera='front', figsize=(14, 7)):
    """
    Draw projected 3D bounding boxes onto an RGB image.

    Parameters
    ----------
    img            : np.ndarray (H, W, 3)
    anns           : list of dict  Annotation list.
    K              : np.ndarray (3, 3)  Intrinsic matrix.
    T_ego_to_sensor: np.ndarray (4, 4)  Ego -> sensor transform.
    camera         : str                Camera name for title.
    figsize        : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    for ann in anns:
        cat    = ann.get('category', 'car').lower()
        color  = CAT_COLORS.get(cat, '#ffffff')
        corners_ego = ego_box_corners_3d(ann)
        corners_uvd = project_ego_to_img(corners_ego, K, T_ego_to_sensor, h, w)

        if (corners_uvd[:, 2] > 0).sum() >= 4:
            _draw_box_edges(ax, corners_uvd, color)
            visible = corners_uvd[corners_uvd[:, 2] > 0]
            if len(visible):
                top = visible[np.argmin(visible[:, 1])]
                if 0 <= top[0] < w and 0 <= top[1] < h:
                    ax.text(top[0], top[1] - 5, cat,
                            fontsize=7, color=color, weight='bold')

    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.6)
    ax.set_title(f'3D bounding boxes — {camera} camera')
    ax.axis('off')
    plt.tight_layout()
    return fig
