# Coordinate Transformations in Griffin: From Scratch to Deep Dive

This document teaches the coordinate mathematics behind the Griffin dataset
starting from absolute zero. You do not need any background in linear algebra,
robotics, or computer vision. We build every idea from the ground up, then go
all the way to the exact transforms used in the code.

If you only remember one sentence from this entire document, make it this one:

> Griffin LiDAR points and all 3D annotations are stored in the **ego frame**
> (the car is at the origin). Projecting them onto a camera goes
> **ego -> sensor -> image** directly, with no vehicle-pose transform.

Everything below explains what that sentence means and why it is true.

---

## Part 1 — The absolute basics

### 1.1 What is a coordinate?

Imagine you are standing in an empty room and someone asks, "where is the
lamp?" You cannot answer with a single number. You have to say something like
"three metres to my right, two metres forward, and one metre up." Those three
numbers — right, forward, up — are a **coordinate**. They locate a point in
space relative to where you are standing and which way you are facing.

A coordinate is always three numbers (in 3D), usually written `(x, y, z)`.
Each number measures distance along one direction.

### 1.2 What is a coordinate frame?

The three numbers only mean something if we agree on:

1. **Where we start measuring from** (the origin), and
2. **Which direction each number counts along** (the axes).

That agreement — an origin plus three axis directions — is called a
**coordinate frame** (or "frame" for short). It is the answer to "three metres
to the right of *what*, and forward in *which* direction?"

Change the frame and the same physical point gets different numbers. The lamp
is at `(3, 2, 1)` relative to you, but relative to someone standing across the
room facing the other way it might be `(-3, -2, 1)`. The lamp did not move.
Only the frame of reference changed.

This is the single most important idea in the whole document. A point in the
real world has no coordinates of its own. It only has coordinates *relative to
a chosen frame*. Most bugs in sensor fusion come from using numbers measured in
one frame as if they were measured in another.

### 1.3 The convention for axis directions

We need names for the three axes. Different fields use different conventions.
Griffin (and most robotics) uses this one for the vehicle:

```
X = forward   (the direction the car is driving)
Y = left      (toward the driver's left)
Z = up        (toward the sky)
```

This is a right-handed frame, which is just a rule about how the three axes are
arranged relative to each other (if you point the fingers of your right hand
from X toward Y, your thumb points along Z). You do not need to worry about
handedness to follow this document; just remember forward-left-up.

---

## Part 2 — Moving between frames

### 2.1 Two operations: translation and rotation

To convert a point's coordinates from one frame to another, you only ever need
two kinds of operation.

**Translation** is a shift. If frame B's origin is 10 metres in front of frame
A's origin, then to convert a point from A's numbers to B's numbers you subtract
10 from the forward coordinate. Translation handles the fact that the two frames
start measuring from different places.

**Rotation** is a turn. If frame B is rotated 90 degrees relative to frame A
(say B faces left where A faces forward), then "forward" in A is "right" in B,
and the coordinates get mixed together. Rotation handles the fact that the two
frames point their axes in different directions.

Any change of frame is some translation plus some rotation. Nothing else is
needed (as long as both frames measure distance in the same units, which they
always do here — metres).

### 2.2 Representing a rotation with a matrix

A rotation in 3D is captured by a 3-by-3 grid of numbers called a **rotation
matrix**, usually written `R`. You do not need to know how the nine numbers are
computed. You only need to know what the matrix *does*: you multiply it by a
point's coordinates and you get the same physical point's coordinates after the
rotation.

```
rotated_point = R @ point
```

The `@` symbol is matrix multiplication. Think of `R` as a machine: feed in a
coordinate, get out the rotated coordinate.

### 2.3 Combining rotation and translation

A full change of frame rotates first, then shifts:

```
point_in_B = R @ point_in_A + t
```

where `t` is the translation vector (the three-number shift). This one line is
the heart of everything. Rotate the point into the new orientation, then move it
by the offset between the two origins.

### 2.4 The trick that makes it one operation: homogeneous coordinates

Carrying around "multiply by R, then add t" is clumsy, especially when you chain
several frame changes. There is a clean trick. We add a fourth number, always
equal to 1, to every point:

```
(x, y, z)  becomes  (x, y, z, 1)
```

This four-number version is called a **homogeneous coordinate**. The reason for
the trick is that now the rotation *and* the translation can be packed into a
single 4-by-4 matrix, usually written `T`:

```
T = | R    t |        (R is the 3x3 rotation, t is the 3x1 translation,
    | 0    1 |         the bottom row is just 0 0 0 1)
```

And the whole frame change becomes a single multiplication:

```
point_in_B = T @ point_in_A
```

That is the entire reason 4x4 matrices appear everywhere in this field. They are
just a tidy way to carry a rotation and a translation together so that chaining
frame changes is plain matrix multiplication.

### 2.5 Going the other way: the inverse

If `T` converts coordinates from frame A to frame B, then the **inverse**,
written `inv(T)` or `T^-1`, converts them back from B to A. In code:

```python
T_B_to_A = np.linalg.inv(T_A_to_B)
```

This matters because calibration files often give you the transform in one
direction when you need the other. Inverting is how you flip it.

### 2.6 Chaining frame changes

If you can go A -> B with `T1` and B -> C with `T2`, then you can go A -> C by
multiplying the matrices:

```
T_A_to_C = T2 @ T1      (note the order: the first transform is on the right)
```

The order looks backwards but is correct: the point gets hit by `T1` first
(rightmost), then `T2`. Read matrix chains right to left.

---

## Part 3 — The frames in Griffin

There are three frames you need to know. Each has a clear physical meaning.

### 3.1 The world frame (ENU)

ENU stands for East-North-Up. It is a fixed frame nailed to the ground:

```
X = East, Y = North, Z = Up
```

Its origin is some fixed reference point in the simulated city. It never moves.
This is the frame you would use to say "the car is at this absolute location on
the map." The vehicle's **pose** (its position and heading) is expressed in this
frame.

### 3.2 The ego frame (the vehicle)

The ego frame is bolted to the car. Its origin is a reference point on the
vehicle, and its axes are forward-left-up as described earlier. Because it is
attached to the car, it moves and turns with the car. From the ego frame's point
of view, the car is always at `(0, 0, 0)` and always facing along +X, and it is
the rest of the world that appears to move.

**This is the frame Griffin stores its LiDAR points and annotations in.** We
will return to this crucial fact.

### 3.3 The camera frame

Each camera has its own frame, attached to its lens. Cameras use a different
axis convention, the OpenCV convention:

```
X = right (across the image)
Y = down  (down the image)
Z = forward (straight out of the lens, into the scene)
```

The reason Z points into the scene is that Z is the **depth** axis, and depth is
what a camera fundamentally measures along. This matters for the projection math
in Part 5.

### 3.4 How the frames relate

- The **pose** (from the pose file) relates the ego frame to the world frame.
  It tells you where the car is and which way it points in the world.
- The **extrinsic** (from each calibration file) relates a sensor frame to the
  ego frame. It tells you where that sensor is mounted on the car and which way
  it faces.

That is the whole map: world `<->` ego via the pose, and ego `<->` sensor via
the extrinsic.

---

## Part 4 — The fact that trips everyone up

### 4.1 Where are the LiDAR points actually stored?

When you open a Griffin LiDAR `.ply` file you get a list of points, each with an
`(x, y, z)`. The obvious question is: **in which frame are these numbers?**

It is tempting to assume they are in the world frame, because the numbers can be
large (ranging over hundreds of metres). For a long time during the development
of this project, that was the assumption — and it was wrong.

The points are in the **ego frame**. The car is at the origin. The large numbers
are simply far-away objects (buildings hundreds of metres down the road), still
measured relative to the car.

### 4.2 How we proved it

The proof is simple and worth repeating, because it is the kind of check that
settles these questions definitively. We compared, for several frames:

- the car's position in the world (from the pose file), and
- the centre of the LiDAR point cloud (the average of all points).

If the points were in the world frame, the cloud would cluster around the car's
world position. It did not. Across many frames where the car drove from world
position 65 metres down to 35 metres, the point cloud centre stayed pinned near
`(0, 0)`. The only way the cloud can stay at the origin while the car moves
through the world is if the points are measured *relative to the car* — that is,
in the ego frame.

### 4.3 Why this matters so much

If you wrongly believe the points are in the world frame, you will "helpfully"
apply the vehicle pose to move them into the ego frame before projecting them
onto a camera. But they are already in the ego frame, so you have transformed
them twice. The result is subtly wrong: points land near objects but slightly
off. This is exactly the bug that caused projected LiDAR to appear shifted from
the bus and the road in early versions of this project. The fix was to stop
applying the pose.

---

## Part 5 — Projecting LiDAR onto a camera image

This is the central task: given a 3D LiDAR point and a camera, which pixel does
it fall on? Now that we know the points are in the ego frame, the pipeline is
short.

### 5.1 The pipeline

```
point in ego frame
    |  T_ego_to_sensor    (the camera extrinsic, inverted)
    v
point in camera frame
    |  K                  (the camera intrinsic; uses depth = Z)
    v
pixel (u, v)
```

Two steps. No pose. The pose belonged to a world-to-ego conversion we do not
need, because we already start in the ego frame.

### 5.2 Step one: ego to camera

The calibration file for a camera stores its **extrinsic**, a 4x4 matrix that is
`T_sensor_to_ego` — it would take a point from the camera frame into the ego
frame. We want the opposite direction (ego into camera), so we invert it:

```python
T_ego_to_sensor = np.linalg.inv(extrinsic)
point_cam = T_ego_to_sensor @ point_ego      # using homogeneous coordinates
```

After this, the point is expressed in the camera's own right-down-forward frame.

### 5.3 Step two: camera to pixel with the intrinsic matrix

The **intrinsic matrix** `K` describes the camera's lens and sensor:

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

- `fx, fy` are the focal lengths in pixels (how strongly the lens magnifies).
- `cx, cy` are the principal point, usually the image centre.

For Griffin's cameras, `fx = fy = 687.29` and `(cx, cy) = (960, 540)` for a
1920-by-1080 image, so the principal point is dead centre.

The projection itself divides by depth. If the point in the camera frame is
`(Xc, Yc, Zc)`:

```
u = fx * (Xc / Zc) + cx        (pixel column)
v = fy * (Yc / Zc) + cy        (pixel row)
```

The division by `Zc` is **perspective**: things twice as far away appear half as
big. This is why Zc must be the depth (forward) axis. If your point has depth in
the wrong slot, this formula produces nonsense, which is the symptom of a wrong
axis convention.

Only points with `Zc > 0` are in front of the camera; points with `Zc <= 0` are
behind it and must be discarded before dividing.

### 5.4 The exact code

```python
def project_lidar_to_image(pts_ego, K, T_ego_to_sensor, img_h, img_w, max_depth=80.0):
    N = pts_ego.shape[0]
    # ego -> camera (homogeneous)
    pts_cam = (T_ego_to_sensor @ np.hstack([pts_ego, np.ones((N, 1))]).T).T[:, :3]
    depth = pts_cam[:, 2]
    keep  = depth > 0.1                 # in front of the camera only
    pts_cam, depth = pts_cam[keep], depth[keep]
    # camera -> pixel
    proj = (K @ pts_cam.T).T
    u = proj[:, 0] / depth
    v = proj[:, 1] / depth
    # keep only pixels inside the image
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (depth < max_depth)
    return np.stack([u[inside], v[inside], depth[inside]], axis=1)
```

Notice there is no pose anywhere. Annotations (bounding boxes) use exactly the
same path, because they are also in the ego frame.

---

## Part 6 — When you DO need the pose: the world-frame viewer

The pose has not become useless. It is needed whenever you want to leave the ego
frame and place data into the world frame. The 3D viewer is the main example.

In the viewer we want to watch the car drive through a fixed scene. That means
everything must be in the world frame: the buildings stay put, the car moves. So
we take the ego-frame LiDAR points and lift them into the world using the pose.

The pose file gives the ego's position and orientation in the world, which is
the transform `T_ego_to_ENU` (ego into world). Our loader returns its inverse,
`T_ENU_to_ego`, so we invert once to get the direction we want:

```python
T_ENU_to_ego, pose = load_pose_griffin(pose_file)   # world -> ego
T_ego_to_ENU = np.linalg.inv(T_ENU_to_ego)          # ego -> world
points_world = (T_ego_to_ENU @ homogeneous(points_ego)).T[:, :3]
```

Now the points sit at the car's actual world location, and as the car drives
from frame to frame the cloud travels with it through the static scene. This is
the difference between the two viewpoints:

- **Ego frame:** the car is fixed at the origin and the world flows past it.
  Best for "what does the car see right now."
- **World frame:** the world is fixed and the car drives through it. Best for
  "watch the trajectory unfold."

Both are correct. They are just different choices of which frame to render in.

### 6.1 Euler angles: how the pose builds its rotation

The pose stores orientation as three angles: roll, pitch, and yaw. These are
**Euler angles** — three successive turns about three axes that together produce
any 3D orientation. Roll is a turn about the forward axis, pitch about the
sideways axis, yaw about the vertical axis (the heading you would read off a
compass).

The catch is that the three turns must be applied in a specific order, and
different software uses different orders. Griffin uses the order `xyz`
(roll about X, then pitch about Y, then yaw about Z), documented in its
`space_utils.py`. Using the wrong order, such as `zyx`, produces a different
rotation matrix and silently corrupts every downstream transform. Our loader
uses the correct order:

```python
R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
```

This was another real bug earlier in the project: an initial `zyx` guess threw
the whole ego frame off by a large angle and pushed projected points off-screen.

---

## Part 7 — Sensor mounting positions

Each sensor sits at a different spot on the car. The LiDAR is on the roof, the
front camera on the windshield, the rear camera on the trunk. Their extrinsics
encode exactly these mounting offsets.

For the LiDAR, the extrinsic translation is roughly `(0.25, 0.00, 1.10)` metres
in the ego frame: a quarter metre forward of the reference point and 1.1 metres
up. That 1.1 metres is the roof height. When the 3D viewer draws each sensor as
a small set of axes at its extrinsic position, the markers appear at their true
physical locations on the car, and they all share the one ego frame. They are
not at the same point, and they should not be — they are different devices in
different places, which is the entire reason extrinsic calibration exists.

---

## Part 8 — How to validate a transform yourself

Never trust a transform you have not sanity-checked. The cheapest check is to
push a known point through it and see if the answer makes physical sense.

Place a test point 10 metres directly ahead of the car, at the car's height. In
the ego frame that is `(10, 0, 0)`. Project it onto the front camera:

```python
import numpy as np
p_ego = np.array([[10.0, 0.0, 0.0]])
p_cam = (T_ego_to_sensor @ np.hstack([p_ego, [[1]]]).T).T[0, :3]
print("camera-frame point:", p_cam)          # Z should be ~ +10 (depth)
u = K[0,0] * p_cam[0]/p_cam[2] + K[0,2]
v = K[1,1] * p_cam[1]/p_cam[2] + K[1,2]
print("pixel:", u, v)                          # should be near the image centre (960, 540)
```

What to expect if the transform is correct:

- The camera-frame Z should be about +10. It is the depth, and the point is 10
  metres away. If Z is negative, your forward axis is flipped. If Z is near zero
  while another axis is 10, your axes are swapped.
- The pixel should land near the image centre `(960, 540)`, because a point
  straight ahead appears in the middle of a forward-facing camera.

If both hold, the transform is sound. If not, the specific way it fails tells
you exactly which axis or sign is wrong. This single test would have caught
every coordinate bug this project hit.

---

## Part 9 — Summary checklist

- A point has no coordinates of its own; only coordinates *in a chosen frame*.
- A change of frame is a rotation followed by a translation, packed into one
  4x4 matrix using homogeneous coordinates.
- Invert a matrix to reverse its direction; multiply matrices to chain frames
  (read right to left).
- Griffin has three frames: world (ENU), ego (the car), and per-sensor camera.
- The pose links world and ego. The extrinsic links ego and sensor.
- **Griffin LiDAR points and annotations are in the ego frame.** Projecting them
  onto a camera is ego -> sensor (invert the extrinsic) -> pixel (intrinsic,
  divide by depth). No pose.
- Use the pose only to move ego-frame data into the world frame, as the 3D
  viewer does to make the car drive through a fixed scene.
- Euler angles need the correct order; Griffin uses `xyz`.
- Validate every transform by projecting a known point and checking the depth is
  positive and a forward point lands at the image centre.
