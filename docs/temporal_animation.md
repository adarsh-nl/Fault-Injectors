# Temporal Animation: A Deep Dive

This document explains how the temporal animation notebook works, starting from
first principles. It assumes no prior knowledge of animation, Matplotlib internals,
or performance optimisation. By the end you should understand not just what each
part does, but why it is built that way.

---

## Contents

1. [What an animation actually is](#1-what-an-animation-actually-is)
2. [The naive approach and why it fails](#2-the-naive-approach-and-why-it-fails)
3. [Matplotlib artists: the key concept](#3-matplotlib-artists-the-key-concept)
4. [The three-phase architecture](#4-the-three-phase-architecture)
5. [Phase 1: Preloading](#5-phase-1-preloading)
6. [Phase 2: Precomputing projections](#6-phase-2-precomputing-projections)
7. [Phase 3: The update loop](#7-phase-3-the-update-loop)
8. [The adaptive layout system](#8-the-adaptive-layout-system)
9. [The two toggle dictionaries](#9-the-two-toggle-dictionaries)
10. [Saving to MP4 and displaying inline](#10-saving-to-mp4-and-displaying-inline)
11. [Memory and performance tuning](#11-memory-and-performance-tuning)

---

## 1. What an animation actually is

An animation is nothing more than a sequence of still images, called frames,
shown one after another fast enough that the human eye perceives smooth motion.
Film runs at 24 frames per second. The Griffin dataset was recorded at 10 frames
per second, meaning the sensors captured a snapshot of the world every 0.1 seconds.

To animate the dataset, we take a range of these recorded frames (say frames 700
to 750, which is 5 seconds of real time) and display them in order. Each frame
shows the camera images and LiDAR data captured at that instant.

The challenge is not the concept. The challenge is doing it fast enough and
without running out of memory.

---

## 2. The naive approach and why it fails

The obvious way to build an animation would be:

```
for each frame:
    load the images from disk
    load the LiDAR from disk
    compute the LiDAR projection
    draw everything
    show it
```

This works, but it is painfully slow. There are two reasons.

**Reason 1: Disk reads are slow.** Reading an image file from disk takes
milliseconds to tens of milliseconds. Reading nine camera images plus a LiDAR
file every frame, for fifty frames, means hundreds of slow disk operations.
The animation would stutter badly.

**Reason 2: Redrawing from scratch is slow.** Every time you call a plotting
function like `imshow` or `scatter`, Matplotlib builds a new internal object,
computes its layout, and prepares it for rendering. Doing this fresh for every
frame wastes enormous effort, because the structure of the plot does not change
between frames; only the data inside it does.

The animation notebook avoids both problems through careful architecture.

---

## 3. Matplotlib artists: the key concept

In Matplotlib, everything you see on a plot is an object called an artist.
An image shown with `imshow` is an artist. A set of scattered points from
`scatter` is an artist. A line, a polygon, a piece of text: all artists.

The crucial insight is that an artist can be created once and then updated
many times. Instead of calling `imshow` every frame, you call it once to create
the image artist, then on each subsequent frame you call a method that swaps in
new pixel data without rebuilding the artist:

```python
# Created once
image_artist = ax.imshow(first_frame)

# Updated every frame after that — fast
image_artist.set_data(next_frame)
```

The same pattern applies to scattered points. A scatter artist stores the point
positions in an internal array. You can replace that array directly:

```python
# Created once
scatter_artist = ax.scatter([], [], c=[])

# Updated every frame — fast
scatter_artist.set_offsets(new_xy_positions)   # set positions
scatter_artist.set_array(new_colour_values)     # set colours
```

`set_offsets` takes an array of shape (N, 2) giving the x and y pixel position
of each point. `set_array` takes the values used to colour each point through
the colour map. Updating these two arrays is far cheaper than calling `scatter`
again.

This single idea, create once and update in place, is the foundation of every
fast Matplotlib animation.

---

## 4. The three-phase architecture

The notebook separates work into three distinct phases, ordered so that all the
slow work happens before the animation begins:

```
PHASE 1 (slow, done once): Preload all frames from disk into memory
PHASE 2 (slow, done once): Precompute all LiDAR projections
PHASE 3 (fast, per frame): Update artists from preloaded data
```

By the time the animation actually runs, every expensive operation is already
finished. The per-frame update does nothing but copy already-computed arrays
into existing artists. This is what makes playback smooth.

---

## 5. Phase 1: Preloading

The preload step reads every image, LiDAR scan, pose, and annotation for the
chosen frame range and stores them in a Python dictionary called `DATA` that
lives in memory (RAM).

```python
DATA = {
    'images': {camera_name: [frame0_img, frame1_img, ...]},
    'pts':    [frame0_lidar, frame1_lidar, ...],
    ...
}
```

After preloading, accessing the image for camera "vehicle_front" at local frame
index 5 is instant:

```python
DATA['images']['vehicle_front'][5]
```

No disk access happens during the animation because everything is already in RAM.

### The downsample parameter

High-resolution images consume a lot of memory. A single 1920 by 1080 colour
image is about 6 megabytes. Fifty frames across nine cameras is over 2 gigabytes,
which can exhaust available RAM.

The `DOWNSAMPLE` parameter shrinks each image before storing it. With
`DOWNSAMPLE = 2`, each image dimension is halved, so a 1920 by 1080 image becomes
960 by 540, using one quarter of the memory. The animation looks slightly less
sharp but uses far less RAM. With `DOWNSAMPLE = 4`, memory drops to one sixteenth.

### Conditional loading

The notebook only loads what it needs. If you have turned off the drone cameras
in the `DISPLAY` dictionary, their images are never read from disk. If no panel
needs the LiDAR, the point clouds are skipped entirely. This is controlled by
flags computed at the top of the preload step:

```python
need_lidar = ('bev' in ACTIVE_PANELS) or (len(OVERLAY_CAMERAS) > 0)
```

This means turning panels off does not just declutter the figure; it also speeds
up loading and reduces memory use.

---

## 6. Phase 2: Precomputing projections

Projecting LiDAR points onto a camera image involves matrix multiplications and
filtering (the full mathematics is covered in `coordinate_transformation.md`).
This is too slow to run live during the animation.

So the notebook computes the projection for every frame in advance and stores the
results:

```python
PROJECTIONS = {
    'vehicle_front': [uvd_frame0, uvd_frame1, ...],
    ...
}
```

Each `uvd` entry is an array where each row is one projected point with three
values: its horizontal pixel position u, its vertical pixel position v, and its
depth (distance from the camera). The depth is used to colour the points so that
near points and far points look different.

This precomputation runs only for cameras that are both displayed and have their
LiDAR overlay turned on. Everything else is skipped.

---

## 7. Phase 3: The update loop

The `update` function is the heart of the animation. Matplotlib calls it once for
each frame, passing the local frame index (0 for the first frame, 1 for the
second, and so on).

The function does only three kinds of work, all of them fast:

**Update the title.** It reads the vehicle pose for this frame and writes the
frame number, timestamp, velocity, and heading into the figure title.

**Update the camera images.** For each displayed camera, it calls `set_data` on
that camera's image artist with the preloaded image for this frame.

**Update the LiDAR overlays and the bird's-eye view.** For each overlay camera,
it calls `set_offsets` and `set_array` on the scatter artist with the precomputed
projection. For the bird's-eye view, it updates the point cloud scatter and
redraws the bounding boxes.

### Why bounding boxes are handled differently

Images and scatter points can be updated in place, but bounding boxes are drawn
as polygon artists, and the number of boxes changes from frame to frame (cars
enter and leave the scene). So the update function removes all the polygons from
the previous frame and creates fresh ones for the current frame:

```python
for patch in bev_box_patches:   # remove old boxes
    patch.remove()
bev_box_patches = []
for ann in annotations:          # add new boxes
    poly = make_polygon(ann)
    ax.add_patch(poly)
    bev_box_patches.append(poly)
```

This is slightly less efficient than in-place updates, but the number of boxes is
small (typically under twenty) so the cost is negligible.

---

## 8. The adaptive layout system

The notebook does not use a fixed arrangement of panels. Instead, it reads the
`DISPLAY` dictionary, collects the panels you turned on, and computes a grid that
fits exactly those panels.

```python
ACTIVE_PANELS = [name for name, on in DISPLAY.items() if on]
n_panels = len(ACTIVE_PANELS)
ncols    = min(MAX_COLS, n_panels)        # at most MAX_COLS per row
nrows    = ceil(n_panels / ncols)         # enough rows to fit them all
```

If you enable three panels, you get a single row of three. If you enable five
panels with `MAX_COLS = 3`, you get two rows (three on top, two below). The panels
are placed into the grid cells in order, with the bird's-eye view always first.

This is why turning a panel off in `DISPLAY` makes the others grow to fill the
space: the grid is recomputed from scratch based on how many panels are active.

---

## 9. The two toggle dictionaries

The notebook has two separate control dictionaries, and understanding the
difference is important.

**`DISPLAY`** decides which panels appear in the figure at all. A panel set to
`False` here is completely absent: not loaded, not drawn, not taking up space.

**`LIDAR_OVERLAY`** decides which of the displayed camera panels get LiDAR points
drawn on top of them. This only has an effect on cameras that are also enabled in
`DISPLAY`. Turning on the overlay for a camera that is not displayed does nothing.

The two are separate because they answer different questions. `DISPLAY` answers
"what do I want to see?" and `LIDAR_OVERLAY` answers "for the things I am seeing,
which should have LiDAR points painted onto them?"

A camera shows its LiDAR overlay only when both conditions are true: it is in
`DISPLAY` as `True`, and it is in `LIDAR_OVERLAY` as `True`.

---

## 10. Saving to MP4 and displaying inline

Once the animation object is built, the notebook does two things with it.

**Saving to MP4.** The `FFMpegWriter` takes each rendered frame and encodes it
into a compressed video file using the ffmpeg program. This requires ffmpeg to be
installed on the system. The resulting MP4 can be shared, embedded in slides, or
played in any video player.

**Displaying inline.** The `to_jshtml` method converts the animation into an
HTML and JavaScript widget with play, pause, and frame-stepping controls that
renders directly inside the Jupyter notebook. This is convenient for quick
inspection without leaving the notebook.

Both come from the same animation object, so the inline preview and the saved
file are identical.

---

## 11. Memory and performance tuning

If the animation runs out of memory or renders too slowly, there are three levers,
in order of impact:

**Reduce the frame count.** Animating frames 700 to 720 instead of 700 to 800
uses one fifth of the memory and renders five times faster. This is the simplest
and most effective control.

**Increase the downsample factor.** Going from `DOWNSAMPLE = 1` to `DOWNSAMPLE = 2`
cuts image memory to one quarter. Going to `DOWNSAMPLE = 4` cuts it to one
sixteenth. The trade-off is image sharpness.

**Reduce the number of panels.** Each panel you turn off in `DISPLAY` means fewer
images to load, hold in memory, and render. Turning off the four side cameras you
do not need is both faster and lighter.

A practical starting point for a memory-constrained machine is twenty frames at
`DOWNSAMPLE = 2` with only the panels you actually want to inspect. Once that
works, increase the frame count until you hit the limits of your hardware.
