# Coordinate Frame Transformations in Griffin

This document is a self-contained reference for the sensor frame mathematics
used in the Griffin dataset. It explains every transform in the projection
pipeline from first principles, documents the specific conventions Griffin uses,
and describes the failure modes encountered when those conventions are wrong.

---

## 1. Why Multiple Coordinate Frames Exist

A LiDAR scanner and a camera are physically separate devices mounted at different
positions on the vehicle. Each sensor measures the world from its own viewpoint.
The LiDAR returns points relative to its own origin. The camera captures pixels
relative to its own optical centre.

Before we can ask "which pixel corresponds to this LiDAR point", we must first
express the LiDAR point in the camera's own coordinate system. This is the
fundamental problem that extrinsic calibration solves.

In Griffin there are three distinct frames involved:

```
ENU (world)  <-->  Ego (vehicle body)  <-->  Camera
```

Each arrow represents a 4x4 rigid body transform (rotation + translation).

---

## 2. The ENU Frame

ENU stands for East-North-Up. It is a fixed global frame:

```
X = East
Y = North
Z = Up
```

Griffin stores LiDAR point clouds in this frame. Every (x, y, z) value in a
`.ply` file is an absolute world position. This is unusual compared to datasets
like KITTI, where LiDAR points are stored in the sensor frame. The practical
consequence is that before projecting any LiDAR point onto a camera image, you
must first undo the vehicle's world position and orientation.

---

## 3. The Ego Frame

The ego frame is attached to the vehicle body and moves with it:

```
X = forward (direction of travel)
Y = left
Z = up
```

Object bounding box annotations are expressed in this frame. When the vehicle
drives forward, the ego X axis points ahead of it. When the vehicle turns left,
the ego Y axis points further left in world space.

The ego frame is right-handed and follows the robotics convention (sometimes
called FLU: Forward-Left-Up).

---

## 4. The Camera Frame

The camera frame is attached to the optical centre of each camera. The OpenCV
convention, which Griffin uses, defines:

```
X = right (in the image plane)
Y = down (in the image plane)
Z = forward (depth, perpendicular to the image plane and pointing into the scene)
```

This convention is required for the standard pinhole projection formula to work:

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

If Z is not the depth axis, this formula cannot be used directly.

---

## 5. Homogeneous Coordinates and 4x4 Transforms

A 3D point p = (x, y, z) is written in homogeneous form as a 4x1 column vector:

```
p_hom = [x, y, z, 1]^T
```

A rigid body transform (rotation R and translation t) is represented as a 4x4
matrix:

```
T = | R   t |   where R is 3x3, t is 3x1
    | 0   1 |
```

Applying the transform to a point is a single matrix multiplication:

```
p_new = T @ p_hom
```

This handles both rotation and translation in one operation. The reason for the
homogeneous representation is that translation cannot be expressed as a linear
(matrix-vector) operation in 3D, but it can be expressed as one in 4D projective
space.

---

## 6. The Pose Transform: ENU to Ego

The pose file for each frame contains:

```json
{
  "x": -77.83,
  "y": -19.91,
  "z":  0.031,
  "roll":  -0.0036,
  "pitch":  0.294,
  "yaw":   89.64,
  "timestamp": 12433
}
```

These describe the vehicle's position (x, y, z) and orientation (roll, pitch, yaw)
in the ENU world frame. In other words, this is the pose of the ego frame
expressed in ENU coordinates. The transform that takes a point from ego frame to
ENU frame is:

```
T_ego_to_ENU = | R_ego_to_ENU   t_ego |
               | 0              1     |
```

where:
- `t_ego = [x, y, z]` (vehicle position in world)
- `R_ego_to_ENU` is the rotation matrix constructed from roll, pitch, yaw

### Euler angle order

The rotation matrix is built from three elementary rotations. The order matters
and must match what the dataset authors intended. Griffin uses:

```python
R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
```

The string `'xyz'` means: rotate around X first (roll), then Y (pitch), then Z (yaw).
These are intrinsic rotations (each rotation is applied relative to the already-rotated
frame). This is confirmed in Griffin's `space_utils.py`:

```python
R_ego_to_ENU = R.from_euler(
    'xyz', [pose_data.roll, pose_data.pitch, pose_data.yaw], degrees=True
).as_matrix()
```

Using `'zyx'` instead (yaw-pitch-roll, the aerospace convention) produces a
different matrix for any frame where all three angles are nonzero. Because
Griffin's yaw values are large (e.g. 89.64 degrees), this error is severe: the
resulting ego frame can be rotated nearly 90 degrees from the correct orientation,
causing almost all LiDAR points to miss the camera field of view entirely.

### Inverting the transform

To project from ENU world into ego frame, we need the inverse:

```
T_ENU_to_ego = inv(T_ego_to_ENU)
```

For a rigid body transform:

```
inv(T) = | R^T   -R^T @ t |
         | 0      1       |
```

where `R^T` is the transpose of R (which equals the inverse for rotation matrices).
In practice, `numpy.linalg.inv` is used directly, which is numerically equivalent.

---

## 7. The Extrinsic Calibration: Sensor to Ego

Each camera's calibration file contains an `extrinsic` field: a 4x4 matrix.
This matrix describes the camera's pose in the ego frame. That is, it transforms
a point from camera coordinates into ego coordinates:

```
T_sensor_to_ego: p_ego = T_sensor_to_ego @ p_cam
```

To project a point from ego space into the camera, we need the inverse:

```
T_ego_to_sensor = inv(T_sensor_to_ego)
```

### Reading the front camera extrinsic

The front camera extrinsic is:

```
[[ 0,  0,  1,  0.357],
 [-1,  0,  0,  0.000],
 [ 0, -1,  0,  1.007],
 [ 0,  0,  0,  1.000]]
```

The translation part `[0.357, 0, 1.007]` tells us the camera is located 0.357 m
in front of the ego origin and 1.007 m above it. This makes physical sense for
a front-mounted camera on the vehicle roof.

The rotation part can be read column by column. Each column of R tells us where
one ego axis points in camera space:

- Ego X (forward) maps to camera column 1: [0, -1, 0] = camera -Y.
  Since camera -Y is "up", ego-forward points up in camera space. This is because
  the camera is mounted looking forward, so objects in front are in the centre
  of the image, not above it. Wait — this is confusing. Let us read it the other way.

Reading row by row: each row of R tells us what component of each ego axis contributes
to one camera axis:

- Camera X (right) = 0*ego_X + 0*ego_Y + 1*ego_Z. The camera's right direction is
  the ego's up direction. This makes sense for a forward-facing camera: right in the
  image corresponds to rightward in the world (which is -Y in ego, or equivalently the
  ego Y axis points left while camera X points right, hence the sign flip).

This rotation encodes the standard relationship between a vehicle's body frame and
a forward-facing camera: what is "depth" to the camera is "forward" to the vehicle,
what is "right" in the image is roughly "right" in the vehicle, and what is "down"
in the image is roughly "down" in the vehicle.

### Why no manual axis swap is needed

An earlier incorrect approach manually permuted the camera-frame coordinates
after applying the extrinsic:

```python
# Incorrect
pts_cv = np.stack([
    pts_cam[:, 2],    # x -> right
    -pts_cam[:, 1],   # y -> down
    pts_cam[:, 0],    # z -> forward
], axis=1)
```

This is wrong because the extrinsic rotation already performs this mapping. The
rotation matrix in `T_sensor_to_ego` encodes exactly the relationship between ego
axes and camera axes. Inverting the matrix gives `T_ego_to_sensor`, which correctly
maps ego coordinates into the camera's [right, down, forward] convention. Applying
an additional manual permutation is equivalent to applying the axis transform twice,
producing a result that is neither the original nor the camera frame.

---

## 8. The Intrinsic Matrix

The intrinsic matrix K maps a 3D point in camera frame to a 2D pixel coordinate:

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

For Griffin's cameras: `fx = fy = 687.29`, `cx = 960`, `cy = 540`.

The projection:

```python
p_img = K @ p_cam        # (3,) vector
u = p_img[0] / p_img[2]  # = fx * (X/Z) + cx
v = p_img[1] / p_img[2]  # = fy * (Y/Z) + cy
```

This only produces valid results when `Z > 0`. Points with `Z <= 0` are behind
the camera and must be filtered before projecting.

The focal lengths `fx` and `fy` are equal, indicating a square pixel sensor.
The principal point `(cx, cy) = (960, 540)` is exactly at the image centre,
indicating no principal point offset (common in simulation).

---

## 9. The Complete Pipeline in Code

```python
from scipy.spatial.transform import Rotation as Rot

def load_pose_griffin(pose_path):
    with open(pose_path) as f:
        p = json.load(f)
    R = Rot.from_euler('xyz', [p['roll'], p['pitch'], p['yaw']], degrees=True).as_matrix()
    T_ego_to_ENU = np.eye(4)
    T_ego_to_ENU[:3, :3] = R
    T_ego_to_ENU[:3,  3] = [p['x'], p['y'], p['z']]
    return np.linalg.inv(T_ego_to_ENU)   # T_ENU_to_ego


def load_calib_griffin(calib_dir, camera='front'):
    with open(os.path.join(calib_dir, f'{camera}.json')) as f:
        cal = json.load(f)
    K               = np.array(cal['intrinsic'],  dtype=np.float64)
    T_sensor_to_ego = np.array(cal['extrinsic'],  dtype=np.float64)
    return K, np.linalg.inv(T_sensor_to_ego)   # T_ego_to_sensor


def project_lidar_to_image(pts_ENU, K, T_ego_to_sensor, T_ENU_to_ego,
                            img_h, img_w, max_depth=80.0):
    N   = pts_ENU.shape[0]

    # Combine: ENU -> ego -> sensor
    T   = T_ego_to_sensor @ T_ENU_to_ego

    # Apply combined transform
    pts_h = np.hstack([pts_ENU, np.ones((N, 1))])
    pts_s = (T @ pts_h.T).T[:, :3]

    # Depth filter
    depth = pts_s[:, 2]
    mask  = depth > 0.1
    pts_s, depth = pts_s[mask], depth[mask]

    # Pinhole projection
    proj = (K @ pts_s.T).T
    u    = proj[:, 0] / depth
    v    = proj[:, 1] / depth

    # Image bounds filter
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (depth < max_depth)
    return np.stack([u[valid], v[valid], depth[valid]], axis=1)
```

---

## 10. Projecting Ego-Frame Points (Annotations)

Object annotations are already in the ego frame. They do not need the pose
transform. The pipeline is shorter:

```
p_ego  ->  T_ego_to_sensor  ->  p_cam  ->  K  ->  pixel
```

```python
def project_ego_to_img(pts_ego, K, T_ego_to_sensor, img_h, img_w):
    N     = pts_ego.shape[0]
    pts_s = (T_ego_to_sensor @ np.hstack([pts_ego, np.ones((N,1))]).T).T[:,:3]
    depth = pts_s[:, 2]
    proj  = (K @ pts_s.T).T
    u = np.where(depth > 0.01, proj[:,0] / np.maximum(depth, 1e-6), -9999)
    v = np.where(depth > 0.01, proj[:,1] / np.maximum(depth, 1e-6), -9999)
    return np.stack([u, v, depth], axis=1)
```

Applying `T_ENU_to_ego` to ego-frame annotations would be incorrect: it would
interpret the annotation's ego-frame coordinates as world coordinates and
transform them into a completely wrong ego position.

---

## 11. Numerical Validation

To verify the pipeline is correct, place a test point at a known location:

```python
# A point 10 m directly in front of the vehicle, at the vehicle's height
# In ego frame: x=10 (forward), y=0, z=0
p_test_ego = np.array([[10.0, 0.0, 0.0]])

p_cam = (T_ego_to_sensor @ np.hstack([p_test_ego, [[1]]]).T).T[0, :3]
print(f"Camera frame: x={p_cam[0]:.3f}  y={p_cam[1]:.3f}  z={p_cam[2]:.3f}")
# Expected: z should be positive and approximately 10.0 (depth = distance)
# x and y should be small (point is roughly centred in front of camera)

u = K[0,0] * p_cam[0]/p_cam[2] + K[0,2]
v = K[1,1] * p_cam[1]/p_cam[2] + K[1,2]
print(f"Projected pixel: u={u:.0f}  v={v:.0f}")
print(f"Image centre:    u={K[0,2]:.0f}  v={K[1,2]:.0f}")
# Expected: pixel should be close to image centre (960, 540) for a forward-facing camera
```

If the projection is correct:
- `z` in camera frame should be approximately 10.0 (positive, equal to the distance)
- The projected pixel should be close to the image centre

If `z` is negative, the depth axis is flipped. If the pixel is far from centre,
the rotation component of the extrinsic is wrong.
