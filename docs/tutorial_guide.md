# Griffin Dataset Visualisation Tutorial Guide

This document describes every cell in the notebook in detail.
It is intended as a reference for understanding not just what the code does, but why
each decision was made — in particular the coordinate frame transformations in Step 5,
which are the most conceptually demanding part.

---

## Contents

1. [Step 0 — Environment Setup](#step-0)
2. [Step 0b — Downloading from Hugging Face](#step-0b)
3. [Step 1 — Dataset Structure](#step-1)
4. [Step 2 — RGB Images: Ground Vehicle](#step-2)
5. [Step 3 — RGB Images: Drone](#step-3)
6. [Step 4 — LiDAR Point Clouds](#step-4)
7. [Step 5 — Calibration and Sensor Fusion](#step-5)
8. [Step 6 — 3D Bounding Box Annotations](#step-6)
9. [Step 7 — Dataset Statistics](#step-7)
10. [Step 8 — Benchmark Results](#step-8)

---

## Step 0 — Environment Setup {#step-0}

The notebook depends on the following libraries:

| Library | Purpose |
|---------|---------|
| numpy | Array operations throughout |
| matplotlib | All 2D plots and image display |
| Pillow | Loading PNG/JPG images |
| plyfile | Reading LiDAR `.ply` point cloud files |
| scipy | Rotation matrix construction from Euler angles |
| plotly | Interactive 3D point cloud viewer in Jupyter |
| open3d | Installed but not used for display (headless servers lack OpenGL) |
| tqdm | Progress bars when iterating over all frames |
| huggingface_hub | Listing and downloading dataset files from the Hub |

All imports are consolidated into one cell at the top so that any missing library
fails immediately rather than halfway through a long computation.

---

## Step 0b — Downloading from Hugging Face {#step-0b}

### Why the dataset is split into zip files

The Griffin dataset is 967 GB total. Rather than requiring a full download,
the authors split each subset into per-modality zip files. This means you can
download only what you need. For the visualisation notebook, the minimum useful
download is:

- `vehicle_metadata.zip` (11 MB) — calibration, labels, poses
- `drone_metadata.zip` (15 MB) — drone calibration and labels
- `vehicle_lidar.zip` (214 MB) — LiDAR point clouds
- `vehicle_camera_front.zip` (16 GB) — front camera images

This totals roughly 16 GB rather than 167 GB for the full 25m subset.

### The extraction path problem

When the zip files extract, they preserve the full internal path. This means
the actual data ends up nested two levels deeper than expected:

```
datasets/griffin_50scenes_25m/griffin-release/
    griffin_50scenes_25m/
        griffin-release/
            vehicle-side/
            drone-side/
```

Rather than hardcoding this path, the notebook uses `rglob('vehicle-side')` to
locate the correct directory automatically regardless of how many levels the zip
adds. This is a robust pattern for any dataset that may change its internal
archive structure between versions.

### The `subsets_map` catalogue

The `list_repo_tree` API call retrieves every file in the `datasets/` folder
with its size. This is stored in `subsets_map` as a dictionary mapping subset
name to a list of `(filename, size_bytes)` tuples. The download plan cell
reads from this dict to validate your file selection before starting any
downloads, so you can see the total size before committing.

---

## Step 1 — Dataset Structure {#step-1}

### Folder organisation

Griffin uses KITTI-style raw data organisation. Each agent (vehicle or drone)
has its own top-level folder with the same internal structure:

```
<agent>-side/
  calib/
  camera/
  lidar/         (vehicle only)
  label/
  pose/
  scene_infos.json
```

### Calibration files

There is one calibration file per sensor, not one per frame. The vehicle has:

- `front.json`, `back.json`, `left.json`, `right.json` — one per camera
- `lidar_top.json` — the LiDAR sensor

Each camera file contains `intrinsic` (3x3 matrix) and `extrinsic` (4x4 matrix).
The LiDAR file contains only `extrinsic`. The meaning of these matrices is
discussed in detail in Step 5.

### Label format

Labels are stored as `.txt` files named by timestamp (e.g. `012433.txt`).
Each line is one object, space-separated:

```
category  x  y  z  l  w  h  roll  pitch  yaw  track_id  visibility
```

Where:
- `x, y, z` is the object centre in the ego (vehicle body) coordinate frame in metres
- `l, w, h` are length, width, height in metres
- `roll, pitch, yaw` are orientation angles in degrees
- `visibility` is a float in [0, 1] representing what fraction of the object is visible
  (Griffin's occlusion-aware metric — see Section 2.2 of the paper)

### Pose files

Pose files are also named by timestamp. Each file is a JSON with fields:
`x, y, z` (position in ENU world frame), `roll, pitch, yaw` (orientation in degrees),
`velocity`, and `timestamp`.

### LiDAR files

LiDAR frames are `.ply` files. The vertex fields are `x`, `y`, `z`, and `I` (intensity).
A key property: these coordinates are in the **ENU (East-North-Up) world frame**, not
the ego or sensor frame. This has important consequences for projection, explained in
Step 5.

---

## Step 2 — RGB Images: Ground Vehicle {#step-2}

The vehicle carries four cameras arranged in a surround configuration:

- Front: faces the direction of travel
- Back: faces rearward
- Left and Right: face sideways

All cameras share the same intrinsic matrix (focal length 687.3 px, principal point
at 960, 540 for a 1920x1080 sensor). The distortion coefficients differ slightly
between cameras because they are physically separate lenses.

Images are displayed in a 2x2 grid in the order front (top-left), right (top-right),
back (bottom-left), left (bottom-right) — a conventional surround-view layout used in
automotive perception.

---

## Step 3 — RGB Images: Drone {#step-3}

The drone carries five cameras. The bottom camera faces straight down and is the
primary perception sensor — it provides a bird's-eye view of the scene below.
The four surround cameras (front, back, left, right) face outward and slightly
downward.

The drone has no LiDAR. This is a deliberate design choice driven by UAV payload
and power constraints. The consequence for perception is that 3D object detection
from the drone side must rely entirely on image-based methods, which are inherently
less geometrically precise than LiDAR-based approaches.

---

## Step 4 — LiDAR Point Clouds {#step-4}

### Loading the point cloud

The `.ply` format stores points as a vertex list. Griffin uses four fields per point:
`x`, `y`, `z` (float32, world frame), and `I` (float32, intensity/reflectance).

The intensity field name is uppercase `I`, not the lowercase `intensity` used in many
other datasets. This is a dataset-specific detail that cannot be inferred from the
format specification — it was discovered by printing `v.data.dtype.names` at runtime.

### Bird's-eye view

Because coordinates are in ENU world frame, the raw x/y range can span hundreds of
metres (e.g. x from -59 to +285). A BEV plot of raw coordinates would show a tiny
cluster somewhere off-centre. The notebook subtracts the ego vehicle's world position
`(cx_ego, cy_ego)` from all points before plotting so that the vehicle is always at
the origin and the scan appears centred.

### Front-view (azimuth vs elevation)

Rather than plotting x vs z (which mixes distance and height in an unintuitive way),
the front-view plots angular coordinates:

- Azimuth: the horizontal angle from straight ahead, computed as `arctan2(y, x)`
- Elevation: the vertical angle, computed as `arctan2(z, sqrt(x^2 + y^2))`

This is the natural coordinate system for a rotating LiDAR. Each horizontal scan
line of the LiDAR corresponds to a fixed elevation angle. The 80 beams of the
Griffin LiDAR appear as 80 roughly horizontal bands in this view.

The plot is filtered to a +/-45 degree horizontal and +/-15 degree vertical field
of view because outside this range the density of points drops sharply and the
structure becomes hard to read.

### Interactive 3D view

Open3D's `draw_geometries` requires a display (OpenGL context), which is not
available on headless Jupyter servers. Plotly is used instead because it renders
directly in the browser as WebGL. Points are subsampled to 20,000 for performance;
the full cloud at ~3,000 points per frame in Griffin does not require subsampling,
but the cap is left in place for use with denser datasets.

---

## Step 5 — Calibration and Sensor Fusion {#step-5}

This is the most technically demanding step. The goal is to project LiDAR points
onto the camera image plane. This requires a precise understanding of three distinct
coordinate frames and two transforms.

### The three coordinate frames

**ENU frame (world frame)**

ENU stands for East-North-Up. This is a global Cartesian coordinate system fixed to
the Earth's surface:

```
X = East
Y = North
Z = Up
```

Griffin stores LiDAR point clouds in this frame. Every point's `x, y, z` value is
an absolute world position, not relative to any sensor or vehicle.

**Ego frame (vehicle body frame)**

The ego frame is attached to the vehicle and moves with it. The convention used in
Griffin is:

```
X = forward (direction of travel)
Y = left
Z = up
```

Object annotations (`x, y, z` in the label files) are expressed in this frame.
When the vehicle turns, the ego frame rotates with it.

**Sensor/camera frame (OpenCV convention)**

The camera frame is attached to the camera lens. The projection formula `K @ p`
only produces correct pixel coordinates when the point is expressed in this frame
with the following axis convention:

```
X = right (in the image plane)
Y = down (in the image plane)
Z = forward (depth, perpendicular to image plane)
```

The key constraint is that Z must be the depth axis because the perspective
projection formula divides x and y by z to produce pixel coordinates:

```
u = fx * (X_cam / Z_cam) + cx
v = fy * (Y_cam / Z_cam) + cy
```

If Z is not the forward axis, this formula produces nonsense.

### The two transforms

**T_ego_to_ENU (pose transform)**

The pose file gives the vehicle's position and orientation in the world:
`x, y, z` (translation) and `roll, pitch, yaw` (rotation in degrees).

This describes where the ego frame is, expressed in the ENU frame. Specifically,
it is the transform that takes a point in ego coordinates and produces its
position in world coordinates:

```
p_ENU = T_ego_to_ENU @ p_ego
```

To go the other direction (world to ego), we invert:

```
T_ENU_to_ego = inv(T_ego_to_ENU)
```

The rotation matrix is constructed from the Euler angles using the convention
`'xyz'` (intrinsic rotations: first roll around X, then pitch around Y, then
yaw around Z). This specific order is documented in Griffin's `space_utils.py`
and differs from the `'zyx'` order used in many aerospace applications. Using
the wrong order produces a completely different rotation matrix and breaks all
downstream projections silently — there is no error, just incorrect results.

**T_sensor_to_ego (extrinsic calibration)**

The camera's `extrinsic` field in the calibration JSON is a 4x4 matrix that
expresses the camera's pose in the ego frame. It is the transform that takes
a point in camera coordinates and produces its position in ego coordinates:

```
p_ego = T_sensor_to_ego @ p_cam
```

To go from ego to camera (which is what we need for projection), we invert:

```
T_ego_to_sensor = inv(T_sensor_to_ego)
```

The LiDAR's `extrinsic` has the same meaning: it describes where the LiDAR
sensor sits in the ego frame. However, because the LiDAR points are already
stored in ENU world coordinates (not in LiDAR sensor coordinates), the LiDAR
extrinsic is not used in the projection pipeline. Only the camera extrinsic
and the pose are needed.

### The complete projection pipeline

Combining the two transforms:

```
1. Start: P_ENU     (LiDAR point in world frame)

2. Apply T_ENU_to_ego:
   P_ego = T_ENU_to_ego @ P_ENU

3. Apply T_ego_to_sensor:
   P_cam = T_ego_to_sensor @ P_ego

4. These two steps can be combined:
   T_ENU_to_sensor = T_ego_to_sensor @ T_ENU_to_ego
   P_cam = T_ENU_to_sensor @ P_ENU

5. Filter: keep only points where Z_cam > 0 (in front of the camera)

6. Apply intrinsic projection:
   u = fx * (X_cam / Z_cam) + cx
   v = fy * (Y_cam / Z_cam) + cy

7. Filter: keep only (u, v) within image bounds [0, W) x [0, H)
```

### Why no axis swap is needed

An earlier version of this notebook applied a manual axis permutation
(swapping columns of the camera-frame point matrix). This was incorrect.

The camera extrinsic rotation matrix already encodes the full mapping from ego
axes to camera axes. Inspecting the front camera extrinsic:

```
[[ 0,  0,  1,  tx],
 [-1,  0,  0,  ty],
 [ 0, -1,  0,  tz],
 [ 0,  0,  0,   1]]
```

Reading the columns: the ego X axis (forward) maps to camera Z (depth).
The ego Y axis (left) maps to camera -X (right in image is -left in ego).
The ego Z axis (up) maps to camera -Y (down in image is -up in ego).

This is precisely the standard camera convention. The rotation is already correct
when we invert the extrinsic. Adding a manual axis swap on top of this applies
the rotation twice and produces wrong results.

### Common failure modes encountered during development

During the development of this notebook, the projection was wrong in several
distinct ways. Understanding each failure is instructive:

**Failure 1: Wrong JSON field names.**
The initial attempt used `cam_intrinsic` and `lidar_to_camera` as key names,
based on analogies with other datasets. The actual field names are `intrinsic`,
`extrinsic`, and `distortion`. The fix was to print the JSON keys before assuming
any structure.

**Failure 2: Wrong extrinsic composition direction.**
The extrinsic was initially treated as a camera-to-world transform and composed
as `T_cam_world @ T_lidar_world`. The extrinsic is actually sensor-to-ego, so
the correct approach is `inv(T_sensor_to_ego) @ inv(T_pose)`. Composing in the
wrong direction produces a matrix that looks plausible but places projected points
in a narrow vertical stripe.

**Failure 3: LiDAR points assumed to be in ego frame.**
The projection was applied directly using extrinsics without first applying the
pose transform. Since the points are in world frame, not ego frame, this produced
projections that were spatially incoherent with the image.

**Failure 4: Wrong Euler angle order.**
The pose was converted using `'zyx'` Euler order instead of the `'xyz'` order
used by the Griffin authors. This produced a rotation matrix that differed from
the correct one by a large angle, causing the ego frame to be misoriented and
placing most points outside the camera field of view.

The correct Euler order was confirmed by reading the Griffin source file
`tools/griffin_data_converter/space_utils.py`, which contains the canonical
implementation of all frame transforms used by the dataset.

### Projecting annotations onto the image

Object annotations are expressed in the ego frame (not the ENU frame), so the
projection for bounding boxes is simpler:

```
1. Start: P_ego   (box corner in ego frame)
2. Apply T_ego_to_sensor only  (no pose step needed)
3. Project with K
```

This distinction matters: using the pose transform on ego-frame annotations
would be incorrect and would produce boxes displaced from the objects.

---

## Step 6 — 3D Bounding Box Annotations {#step-6}

### Box parameterisation

Each Griffin annotation is a 9-DoF box:

- Centre position (x, y, z) in ego frame
- Dimensions (l, w, h) in metres
- Orientation (roll, pitch, yaw) in degrees

The 8 corners of the box are computed by starting from a unit box centred at
the origin, rotating by the yaw angle (the dominant orientation for ground
vehicles), and translating to the centre position. Roll and pitch are near zero
for ground vehicles but nonzero for drones.

### BEV visualisation

The BEV draws the footprint of each box (the bottom 4 corners projected onto
the ground plane) as a filled polygon. The LiDAR scan is drawn underneath in
greyscale as a spatial reference. This view is useful for verifying that
annotation positions are consistent with the point cloud geometry.

### Camera projection

The 3D box corners are projected onto the camera image using `T_ego_to_sensor`
and `K`. Edges between corners are drawn only if both endpoints have positive
depth (Z > 0 in camera frame), avoiding lines that would cross the image plane
and wrap incorrectly.

### Counting annotations

The annotation count cell iterates over all label files in the subset and
accumulates per-category counts. This is a live count from the actual data
rather than a number quoted from the paper, so it reflects whatever was
actually downloaded.

---

## Step 7 — Dataset Statistics {#step-7}

The charts in this step are reproduced from the Griffin paper (Table 1 and
supplementary material). They are not computed from the downloaded data but
are useful as reference when comparing subsets or designing experiments.

**Subset sizes.** Griffin-25m has 47 scenes and approximately 7,050 frames.
Griffin-Random has 104 scenes and approximately 15,600 frames and is the most
challenging subset because the drone altitude varies, making it harder for
models to adapt to a single perspective.

**Altitude distribution.** Each fixed-altitude subset has a tolerance of +/-2 m.
The Random subset samples uniformly between 20 and 60 m. Altitude matters
because higher drones produce smaller apparent object sizes, which degrades
detection performance for all methods.

**Weather and lighting.** The dataset uses CARLA + AirSim co-simulation to
generate photorealistic environments under varied weather (clear, rainy, foggy)
and time of day (noon, sunset, night). This diversity is intentional: a model
that only works in clear daylight is not suitable for real deployment.

---

## Step 8 — Benchmark Results {#step-8}

### Metrics

**AP (Average Precision)** measures detection quality. It summarises the
precision-recall curve across IoU thresholds and object categories. Higher
is better.

**AMOTA (Average Multi-Object Tracking Accuracy)** extends AP to tracking.
It penalises identity switches, missed detections, and false positives across
time. Higher is better.

### Methods

The seven methods span a spectrum of fusion strategies:

| Method | Fusion strategy | Notes |
|--------|----------------|-------|
| No Fusion | Single agent (vehicle only) | Baseline; no cooperation |
| Early Fusion | Raw feature concatenation | Upper bound; requires full bandwidth |
| V2X-ViT | Intermediate, attention-based | Compresses features before sharing |
| Where2Comm | Intermediate, spatial selection | Only transmits high-confidence regions |
| CoopTrack | Instance-level (track sharing) | Shares detected tracks, not raw features |
| UniV2X | End-to-end learned | Joint optimisation of compression and fusion |
| Late Fusion | Decision-level | Merges final detections |

### Key findings

**Altitude sensitivity.** All methods degrade as the drone flies higher. Early
Fusion degrades most sharply (from AP 0.607 at 25m to 0.483 at 55m) because
the raw feature alignment breaks down at large scale changes. CoopTrack and
instance-level methods are more resilient because they share abstract
representations rather than raw pixel or voxel features.

**Communication efficiency.** Early Fusion achieves the highest AP but requires
311 MB/s of bandwidth. Late Fusion requires only 1.6 kB/s but barely improves
over the no-fusion baseline. The intermediate methods (V2X-ViT, Where2Comm,
UniV2X) trade AP for bandwidth reduction. The performance-vs-bandwidth chart
in the notebook visualises this Pareto frontier.

**Occlusion-aware labels.** Training without visibility filtering (using
all annotations regardless of occlusion severity) degrades all models. The
effect is largest for the vehicle-side model, which has more severe occlusions
because it operates at ground level.
