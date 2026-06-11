# Griffin Visualisation Tutorial Guide

This guide walks through every step of the visualisation notebook and explains
both what each part does and why. For the full first-principles treatment of the
coordinate mathematics, read `coordinate_transformation.md` alongside this.

The one fact that shapes everything: **Griffin LiDAR points and 3D annotations
are stored in the ego frame** (the car is at the origin). This determines how we
plot point clouds, how we project onto cameras, and how the 3D viewer works.

---

## Repository layout

```
griffin-visualisation/
  src/
    data_loaders.py    file I/O: images, LiDAR, poses, calibration, labels
    transforms.py      projection and coordinate transforms
    visualisation.py   plotting helpers
    __init__.py        re-exports everything
  notebooks/
    quick_start.ipynb              minimal: load a frame and visualise it
    griffin_visualisation_tutorial.ipynb   the full walkthrough
    temporal_animation.ipynb       multi-camera animation over a sequence
    viser_viewer.py                interactive 3D viewer (world frame)
  docs/
    tutorial_guide.md              this file
    coordinate_transformation.md   coordinate maths from scratch
    animation_deep_dive.md         how the animation is built
```

The `src` package is the single source of truth. Every notebook imports from it,
so the transforms are identical everywhere by construction.

---

## Step 0 — Environment and imports

Installs the scientific stack (numpy, matplotlib, Pillow), the LiDAR reader
(plyfile), the 3D maths helper (scipy), the interactive plot library (plotly),
and the Hugging Face client. The imports cell adds the parent directory to the
path so `from src import ...` works from inside the notebooks folder.

---

## Step 1 — Dataset structure and paths

Griffin uses KITTI-style raw folders. Each agent (vehicle, drone) has `calib`,
`camera`, `label`, `pose`, and the vehicle additionally has `lidar`. Key format
facts:

- Calibration is one file per sensor (not per frame): `front.json`, `back.json`,
  `left.json`, `right.json`, and `lidar_top.json`. Each camera file has an
  `intrinsic` (3x3) and an `extrinsic` (4x4, sensor-to-ego). The LiDAR file has
  only an `extrinsic` (its mounting pose).
- Labels are `.txt`, one file per timestamp, space-separated:
  `category x y z l w h roll pitch yaw track_id visibility`. Positions are in
  the ego frame, angles in degrees, visibility in `[0, 1]`.
- Poses are `.json` per timestamp: `x y z roll pitch yaw velocity timestamp`,
  giving the vehicle's position and heading in the world (ENU) frame.
- LiDAR is `.ply` with fields `x, y, z, I` (intensity, uppercase). **These
  points are in the ego frame.**

`get_file_lists` builds sorted lists of all these paths in one call.

---

## Step 2 — Vehicle cameras

Four wide-angle cameras (front, back, left, right) at 1920x1080, 10 fps. They
share the same intrinsic (focal length 687.29, centre at 960, 540) but have
slightly different distortion coefficients because they are physically separate
lenses. The helper `plot_surround_cameras` lays them out front/right/back/left.

---

## Step 3 — Drone cameras

Five cameras (front, back, left, right, bottom). The bottom camera looks
straight down and is the drone's primary perception sensor. The drone carries no
LiDAR, a deliberate choice driven by UAV payload limits, which is why drone-side
3D perception must be image-based.

---

## Step 4 — LiDAR point clouds

`load_lidar` reads the `.ply` and returns an `(N, 4)` array of `x, y, z,
intensity` in the **ego frame**. Because the car is already at the origin, the
plotting helpers use the raw coordinates with no centring:

- `plot_bev` draws the bird's-eye view (looking straight down), coloured once by
  height and once by intensity, with a marker at the origin for the car.
- `plot_front_view` converts points to azimuth (horizontal angle) and elevation
  (vertical angle) and plots that angular view, which is the natural way to see
  what a spinning LiDAR captured.
- The plotly cell gives an interactive 3D scatter you can rotate in the browser.

A note on Griffin's LiDAR: it is sparse (around 2,900 points per frame) with a
roughly 4 metre minimum range, both properties of the simulated sensor. Do not
expect the dense rings of a real 64- or 128-beam unit.

---

## Step 5 — Calibration and sensor fusion

This is the heart of the tutorial. The goal is to draw LiDAR points onto a
camera image. Because the points are in the ego frame, the pipeline is two
steps:

```
point_ego  ->  T_ego_to_sensor  ->  K  ->  pixel
```

`load_calib_griffin` returns the intrinsic `K` and `T_ego_to_sensor`, which is
the inverse of the stored extrinsic (the file stores sensor-to-ego; we want
ego-to-sensor). `project_lidar_to_image` then transforms each point into the
camera frame, keeps the ones in front of the camera, divides by depth to get
pixels, and keeps the ones inside the image. There is no vehicle-pose step,
because the points never left the ego frame.

`plot_fusion` shows the raw image beside the image with points overlaid and
coloured by depth. Switch the `CAMERA` variable to project onto a different
camera.

For the full reasoning, including the common mistakes (double-transforming with
the pose, wrong Euler order, wrong axis convention), see
`coordinate_transformation.md`.

---

## Step 6 — 3D bounding boxes

Annotations are in the ego frame, so they use the same projection path as the
LiDAR. `ego_box_corners_3d` builds the eight corners of a box from its centre,
dimensions, and yaw. Two visualisations follow:

- `plot_bev_with_boxes` draws the box footprints over the LiDAR bird's-eye view.
- `plot_boxes_on_image` projects the eight corners onto the camera with
  `project_ego_to_img` (which keeps all points so the twelve edges can be drawn)
  and connects them.

The boxes always projected correctly even while the LiDAR overlay was buggy in
early versions, precisely because the boxes used the ego-direct path while the
LiDAR was wrongly being run through the pose. That mismatch was the clue that
eventually revealed the LiDAR is in the ego frame.

A final cell counts annotations per category across the whole subset, a live
count from the data rather than a number quoted from the paper.

---

## Step 7 — Dataset statistics

Charts reproduced from the Griffin paper: scenes and frames per subset, and the
weather and time-of-day distribution. These are reference figures, not computed
from your download. The Random subset is the largest and hardest because the
drone altitude varies, forcing models to handle many viewpoints.

---

## Step 8 — Benchmark results

Average Precision (detection quality) and AMOTA (tracking quality) for several
cooperative-fusion methods across the altitude subsets, from the paper's Table 3.
The headline findings: all methods degrade as the drone flies higher, and there
is a steep trade-off between accuracy and the communication bandwidth each method
needs.

---

## The other notebooks

**quick_start.ipynb** is the minimal path: import `src`, load one frame, and call
the fusion, BEV, and box helpers. Use it as a template for your own scripts.

**temporal_animation.ipynb** animates several cameras and a BEV over a frame
range. Two dictionaries control it: `DISPLAY` (which panels appear, layout
adapts) and `LIDAR_OVERLAY` (which displayed cameras get LiDAR drawn on top).
Because the vehicle LiDAR lives in the vehicle ego frame, overlay is supported on
vehicle cameras only. See `animation_deep_dive.md` for the architecture.

**viser_viewer.py** is the interactive 3D viewer. It renders in the world frame,
so the car drives through a fixed scene with its LiDAR scan and sensor markers
moving along with it. This is the one place the pose is used: ego-frame points
are lifted into the world with `ego_points_to_world` so the scan sits at the
car's true world location each frame. A Playing toggle animates the sequence and
the camera can follow the car.

---

## The mental model to keep

- Plotting LiDAR or boxes on their own, or onto a camera: stay in the ego frame,
  no pose.
- Placing things into a fixed world map (the 3D viewer): lift from ego to world
  with the pose.
- When a projection looks wrong, suspect a frame mismatch first, and validate
  with a known test point (Part 8 of `coordinate_transformation.md`).
