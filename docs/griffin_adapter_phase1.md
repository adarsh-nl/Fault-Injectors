# Griffin adapter — Phase 1 design (routing-spec version)

**Status: design only, awaiting approval.** Nothing implemented, no sweep, no
OpenCOOD-path change, no git. Supersedes the injector-set sections of
`griffin_adapter_scope.md` (whose premise included PoseError; that is now
dropped by spec, which moots the baked-pkl pose seam — see §4).

Every claim below marked *verified* was checked live on staged Griffin data
(`griffin_50scenes_25m`) in `.venv-hpc` (py3.13; `plyfile`/`PIL`/`scipy` all
present — this is the Griffin working env; `opencood-official` stays the
OpenCOOD env and is never used for Griffin).

---

## 1. Model check — no metric exists; Phase 2 is adapter + gates ONLY

There is **no runnable Griffin cooperative model or eval on this machine**.
The released checkpoints and their eval logs are staged, but the codebase
(`AgileDataset`, `projects.mmdet3d_plugin`), the nuScenes-format conversion,
the info pickles, the drone track-query cache, and the py3.8/torch-1.9.1 env
that produced them are all absent (established in `griffin_adapter_scope.md`
§1). **The user will train a Griffin model later.**

Therefore, explicitly: **Phase 2 delivers the adapter, per-agent routing, and
login-node gates. It delivers NO AP-degradation measurement.** Nobody should
expect a metric from this work until a trained Griffin model exists; at that
point the corrupted canonical samples (or a corrupted export) plug into
whatever training/eval pipeline exists then.

## 2. Griffin → canonical mapping

Griffin already normalises into the canonical form — `GriffinDataset`
(`src/datasets/griffin.py`) is a `BaseDataset` producing `CooperativeSample`s
natively. *Verified live, frame 0 of `griffin_50scenes_25m`:*

```
vehicle  type=vehicle  ego=True   pose=True  lidar=(2929, 4)  images=[back,front,left,right]
drone    type=drone    ego=False  pose=True  lidar=None       images=[back,bottom,front,left,right]
```

| Griffin source | canonical destination | notes |
|---|---|---|
| `pose/*.json` (x,y,z,roll,pitch,yaw deg, ENU) | `AgentFrame.pose` = `T_agent_to_world` | scipy Euler `'xyz'` → 4×4, one-way |
| `lidar/lidar_top/*.ply` (x,y,z,**I**) | `AgentFrame.lidar` (N,4) float32, ego-body frame | mount `T_lidar_to_ego` applied by `load_lidar`; **real intensity [0.3175, 0.9981]**, read from the literal `I` field (the corrected `pcd.py` is not even involved — Griffin is PLY, not PCD) |
| `camera/{front,…}/*.png` | `AgentFrame.images[cam]` uint8 RGB | camera names are **explicit whitelists** (`_VEH_CAMERAS`/`_DRONE_CAMERAS`), so the `instance_*` segmentation-mask directories are structurally excluded from injection |
| `calib/*.json` | `AgentFrame.cameras[cam]` (K, T_cam_to_agent) | drone's LiDAR *calib stubs* exist with no data — modality keys off **data presence**, never calib presence |
| `label/*.txt` | `AgentFrame.labels`, `frame='agent'` | per-agent frames (unlike OpenCOOD's shared world dict) — already expressible |
| drone side | `agent_type='drone'`, `lidar=None` | no `lidar/` dir in any subset (verified all four) |

**Round trip.** Unlike OpenCOOD there is **no derived quantity to recompute**:
no `transformation_matrix` is baked anywhere in the `griffin-release` layout,
and with PoseError dropped no injector touches a pose at all. A null routing
pass is therefore a **pure pass-through** — every field bit-identical, trivially
— and that is exactly what the Phase 2 round-trip gate asserts. The one place
bit-identity would be at risk is a future *export-to-disk* path (ego-frame
points → inverse mount → sensor-frame PLY accumulates float error ~1e-15);
that path is **not in Phase 2 scope** and its frame convention is deferred to
when the training pipeline's input format is known.

## 3. Per-agent routing (the spec, restated as the implementation contract)

*Verified live* — the shared gate (`src/adapters/modality.py`, untouched)
already produces the right scopes on a real mixed sample:

```
agents_supporting(sample, 'lidar')  -> ['vehicle']
agents_supporting(sample, 'images') -> ['drone', 'vehicle']
require_all(sample, 'lidar')        -> raises (drone lacks it)   # why Griffin never uses require_all
require_any(sample, 'lidar')        -> passes
require_any(sample, 'images')       -> passes
```

| injector | targets | mechanism |
|---|---|---|
| PointsReduce / LidarFog / LidarSnow | `agents_supporting(s,'lidar')` → vehicle only | drone excluded *by absence from the target list*, never by running on an empty cloud |
| camera injectors (brightness/darkness/fog/snow-image/occlusion) | `agents_supporting(s,'images')` → drone + vehicle | |
| MissingModality (Bernoulli) | drone (camera gate `p_drop_rgb`; `p_drop_lidar` hard-0 — inverse of OpenCOOD's assertion) | note: dropping the camera-only drone's cameras is behaviourally equivalent to dropping the agent; routed as spec'd, logged as `missing_modality` |
| CommLatency | drone + vehicles (modality-agnostic) | frame-pairing stage, §4 |
| AgentDrop | vehicles by default — **see flag below** | membership removal, ego protected |
| PoseError | **not wired** | absent from the Griffin spec entirely, not merely disabled |

**Explicit skip, never silent no-op — the loophole, closed by design:**
every stage logs one row per frame of the form
`targets=[vehicle]; skipped=[drone:no-lidar]`, and two hard cases are logged
skips rather than silent passes:

1. **no-target**: if `agents_supporting` returns empty for a stage, the log
   records `no_target_agents` for that frame. The stage did not run, and the
   log says so — auditable, countable, distinguishable from "ran with no
   visible effect".
2. **empty cloud**: an agent whose cloud is `(0, C)` (e.g. after
   MissingModality) is logged `skipped=[<id>:empty-cloud]`, not fed to
   fog/snow. (`_has` treats an empty array as "has LiDAR" — presence-of-array,
   not presence-of-points — so the router must make this distinction itself.)

**FLAG — AgentDrop's default target set is empty on Griffin.** Every Griffin
scene has exactly **one vehicle and one drone**, and the vehicle **is the ego**
(`ds.ego_id == 'vehicle'`, verified). "Vehicles by default" + ego protection ⇒
**zero droppable agents in every frame** — AgentDrop would log `no_target_agents`
for the entire dataset. The only agent whose loss is expressible on Griffin is
the drone (loss of the aerial cooperator's link). Recommendation: make the
drone droppable; awaiting your confirmation as requested in the spec.

## 4. Fire points per used injector

With PoseError dropped, **no used injector needs the pkl-baked transforms
perturbed** — confirmed stage by stage: LiDAR faults touch points only; camera
faults touch images only; MissingModality touches images only; AgentDrop
touches sample membership; CommLatency changes *which frame is read*, and the
stale frame's pose comes from that frame's own `pose/*.json` — genuine
staleness, nothing baked, nothing recomputed. The baked-pkl seam problem from
`griffin_adapter_scope.md` §5 is **moot** and stays unsolved by design.

| injector | fire point | no-op risk avoided |
|---|---|---|
| LidarFog / LidarSnow | canonical `AgentFrame.lidar` (ego-body frame), **with `T_lidar_to_ego` from `calib/lidar_top.json` passed** so ranges are measured from the sensor origin | omitting the mount transform biases every range by ~1.1 m (z); both injectors already accept the parameter |
| PointsReduce | same array (frame-agnostic subsampling) | blocked until its approved RNG fix lands — same status as OpenCOOD |
| camera injectors | `AgentFrame.images[cam]` — raw uint8 straight from PNG, upstream of any future model preprocessing | our seam is the data layer; a future pipeline consumes what we hand it, so there is no post-bake consumer to miss. `instance_*` masks are outside `images` by whitelist |
| MissingModality | per-frame Bernoulli on the drone's `images` dict | logged with the drawn gate `m_rgb`, so drone-dark frames are auditable |
| CommLatency | frame-index selection at sample assembly (`CommLatencyInjector.apply(dataset, k)` — already works against any `BaseDataset`, Griffin included) | **scene-boundary clamping required**: `GriffinDataset` treats a subset side as one flat sequence (7 000 frames ≈ 50 scenes × ~140), so an unclamped stale index would pair the vehicle with a *different scene's* drone frame. `scene_infos.json` carries `begin_frame` + per-scene `frames` lists (verified) — Phase 2 derives per-scene extents and clamps `k_min` to the scene start, mirroring OpenCOOD's `timestamp_index` clamp |
| AgentDrop | sample membership after assembly, ego-protected | pending the drone-droppability decision |

## 5. Fog/snow on real Griffin clouds — *verified, natively injecting*

Frame `011185`, vehicle LiDAR, loaded through `load_lidar` (mount applied),
`T_lidar_to_ego` passed to both injectors, sev 2, seed 1000:

```
cloud: (2929, 4)  I [0.3175, 0.9981]  mean 0.9376
fog  sev2: pts 2929->2929  meanI 0.9376->0.7650  xyz_moved=916   I_changed=2929/2929
snow sev2: pts 2929->541 (2.3 s)  meanI 0.9376->0.5916
           unchanged=242  attenuated=159  scatter=140  removed=2388
```

The intensity operations find real intensity (the literal `I` PLY field — the
`pcd.py` rgb-unpack fix is not even on this code path). Earlier fog sev 1/3
numbers: meanI → 0.7998 / 1.0321, xyz_moved 820 / 1207. Expected shape for
Phase 2 gates: fog changes every intensity and moves a severity-increasing
fraction of points at constant count; snow attenuates + scatters + removes.
Snow's 82 % removal on this sparse 25 m-altitude cloud is aggressive — the
Phase 2 gate will print per-severity removal fractions so the parameterisation
is reviewed with numbers on the table, not assumed.

## 6. OpenCOOD isolation proof

**Files Phase 2 adds (new, imported by nothing in the OpenCOOD path):**

| file | role |
|---|---|
| `src/adapters/griffin.py` | `GriffinAdapter` + the per-agent router (`FaultSpec`-style, targets via `agents_supporting`, skip-logging) |
| `tools/test_griffin_gates.py` | login-node gates: null round-trip bit-identical, per-agent injection-fires, skips-are-logged, ego protection |

**Files touched: none.** Zero edits to `src/adapters/opencood.py`,
`src/adapters/runtime.py`, `src/adapters/modality.py`, `src/fault_injectors/*`,
`src/datasets/base.py`, or `tools/fi_inference.py`. Even the optional
`src/adapters/__init__.py` re-export is skipped in Phase 2 (`griffin.py` is
imported directly by its gate script), so the diff against the OpenCOOD path
is empty by construction, not by care.

Why the gate needs no change (the earlier scope's flagged item is already
resolved): the per-agent semantics were built into `modality.py` when it was
extracted — Griffin consumes `agents_supporting`/`require_any`, OpenCOOD keeps
`require_all` byte-identical (verified then; re-verified live above on a real
mixed sample). The injectors need no change: fog/snow already take
`T_lidar_to_ego`, MissingModality already exposes the per-gate draw, AgentDrop
already takes protection sets, CommLatency already works against any
`BaseDataset`.

The verified baselines (V2X-ViT 0.84/0.62, CoBEVT 0.85/0.66, Where2comm) run
entirely inside `~/opencood-official` + `~/v2xvit-official` + the wrapper
files above — no Griffin file is on any of those import chains.

---

## Phase 2 record (implemented after approval; drone confirmed droppable)

Files added: `src/adapters/griffin.py` (`GriffinFaultSpec`,
`FaultedGriffinDataset` — latency-aware assembly + per-agent router) and
`tools/test_griffin_gates.py`. Files touched: **none** (mtime-corroborated;
the `lidar_snow.py`/fog-table modifications in `git status` are the earlier
approved fog/snow task, 02:26–02:28 vs Phase 2's 03:06+). OpenCOOD Gate 1
re-run post-implementation: 51 checks, 0 failures.

**Gates: 34 checks, 0 failures** (`test_griffin_gates.py`, real
`griffin_50scenes_25m` data):

- **A** null round trip bit-identical (pure pass-through, no translation layer);
- **B** per-agent fires: fog → vehicle cloud only (drone logged
  `no-lidar`, images untouched); brightness → both agents' images
  (vehicle cloud untouched, per-camera independent seeds); MissingModality
  p=1 → drone cameras zeroed, vehicle untouched; AgentDrop p=1 → drone
  removed, ego kept, `vehicle:ego-protected` logged;
- **C** skips-not-noops: all-lidar-less → `no_target_agents;skipped=[...]`
  **with reasons** (a gate run caught the reasons being dropped in the
  no-target case — fixed); empty `(0,4)` cloud → logged
  `vehicle:empty-cloud`, injector not run;
- **D** ego protection: vehicle undroppable, ego never latency-shifted;
- **E** scene clamp: at the first frames of scene 2, stale draws of up to
  5 frames clamp to the scene start (3/3 boundary cases clamped);
  mid-scene frames shift normally (209→203); deterministic per (seed, k).

**F — snow parameterisation (the review numbers requested):**

```
sev1: 2952 -> 455   REMOVED 84.6%   attenuated=203  scatter=123  meanI 0.9396->0.4111
sev2: 2952 -> 503   REMOVED 83.0%   attenuated=147  scatter=141  meanI 0.9396->0.5006
sev3: 2952 -> 506   REMOVED 82.9%   attenuated=135  scatter=142  meanI 0.9396->0.5168
```

The suspicion was right, and it is worse than "too aggressive": removal is
**severity-insensitive and non-monotonic** (sev1 removes slightly *more* than
sev3). The range-adaptive noise floor dominates on these sparse 25 m clouds,
so the default MultiCorrupt severity mapping yields a flat curve — three
severities that are operationally the same fault. Snow must not enter a
Griffin sweep as-is; recalibration options (retune `noise_floor`, gate the
removal stage, or treat snow as a single binary heavy-corruption condition)
are a design decision for review, not something to pick silently.

**STOP** — gates pass; awaiting snow-parameterisation review before any
Griffin sweep planning.
