# Griffin adapter — Phase 1 scope report

**Status:** scoping only. No adapter written, no sweep, no OpenCOOD code touched,
no git. **Awaiting approval.**

**Headline:** the released Griffin cooperative model is **camera-only**. That
inverts the expected injector set — the LiDAR injectors this task expected to
gain (fog, snow) provably corrupt Griffin *data* but cannot move Griffin
*metrics*, and there is **no runnable Griffin eval on this machine** at all.
Details in §1 and §3. Two findings contradict premises I was given; both are
flagged with evidence rather than quietly adopted.

*(Unrelated: PoseError PoC job 557236 is still running on hpc-node12. Not
forgotten — its monitor is live and I report it when it lands.)*

---

## 1. Is there anything to evaluate? — **Checkpoints yes, runnable eval NO**

### What exists

| item | status | path |
|---|---|---|
| Dataset, 4 subsets | **staged, 994 GB** | `/datasets/eemcs/ps/cv/huggingface/griffin/datasets/` |
| Trained checkpoints | **staged, ~5.3 GB** | `.../griffin/ckpts/` |
| Published eval logs | **staged** | `.../ckpts/<subset>/logs/*-eval.log` |
| Pretrained backbone | staged | `ckpts/bevformer_tiny_epoch_24.pth` |

Per subset: `cooperative/instance_fusion/iter_*.pth`, `early-fusion/`,
`vehicle-side/`, `drone-side/`, plus train/eval logs for cooperative,
vehicle-side, drone-side and no-fusion.

All four subsets are complete — `griffin_100scenes_random` (15 496 ply),
`griffin_50scenes_25m` (7 000), `griffin_50scenes_40m` (8 046),
`griffin_50scenes_55m` (7 450). *(An earlier pass of mine reported 55m as empty;
that was a path-nesting artifact — it nests one level shallower than the others
and its directory is mode `drwx--S---`. It is fully staged. Correcting so nobody
plans around a wrong inventory.)*

### The published oracle — `griffin_50scenes_25m`, cooperative

From `ckpts/griffin_50scenes_25m/logs/cooperative-eval.log`:

```
pts_bbox/mAP   0.1442      pts_bbox/NDS   0.1972
car_AP_dist_0.5  0.1614    car_AP_dist_1.0  0.2931
car_AP_dist_2.0  0.4879    car_AP_dist_4.0  0.7042
AMOTA 0.1452   AMOTP 1.7099   MOTA 0.1282   recall 0.1814
```

Note this is a **nuScenes-style centre-distance mAP/NDS plus AMOTA tracking**
metric — *not* IoU-based AP@0.5/0.7. It is not comparable in kind to the
OpenCOOD baselines, and a "Griffin AP drop" is a different quantity from a
"CoBEVT AP drop". Worth settling before any cross-dataset table gets built.

### What is missing — this is the blocker

The eval that produced those numbers cannot be run here. Missing:

1. **The Griffin codebase.** `dataset_type = 'AgileDataset'`, `projects.mmdet3d_plugin`.
   No clone anywhere on this machine.
2. **The converted dataset.** Config reads
   `./datasets/agile_50scenes_25m/agile-nuscenes/cooperative/` — a
   **nuScenes-format conversion**. We have only the raw `griffin-release/`
   layout. Nothing named `agile*` exists under the mount.
3. **Info pickles.** `agile_infos_train.pkl` / `agile_infos_val.pkl` and
   `split_datas/agile_50scenes_25m.json`. Zero `.pkl` files staged.
4. **The drone-side track-query cache.** `LoadInfInformation` reads
   `drone-side/track_query/` for `query_feats`, `query_embeds`, `obj_idxes`,
   `ref_pts`. Not staged.
5. **The environment.** Training ran Python 3.8.20, torch 1.9.1+cu111,
   mmdet/mmdet3d, env `v2xtrack`. No mmdet/mmcv/mmengine in any conda env here
   (`thesis` torch 2.5.1, `dhd_env` 2.0.1, `opencood-official` 1.12.1,
   `.venv-hpc` 2.13 — none has the mm stack).

**Plainly: Griffin checkpoints and published numbers exist; the pipeline to
reproduce them does not.** Items 2–4 are *generated artifacts* — reproducing
them means running Griffin's own converter and its drone-side inference pass,
not just downloading more data.

---

## 2. Griffin's data model vs OpenCOOD's

### On-disk layout (per subset, per side)

```
griffin-release/
  vehicle-side/   calib/{front,back,left,right,lidar_top}.json
                  camera/{front,back,left,right}/*.png  + instance_*/  (masks)
                  lidar/lidar_top/*.ply
                  label/*.txt      pose/*.json      scene_infos.json
  drone-side/     calib/{front,back,left,right,bottom, lidar_bottom,
                         lidar_front, lidar_virtual}.json
                  camera/{front,back,left,right,bottom}/*.png + instance_*/
                  label/*.txt      pose/*.json      scene_infos.json
                  --- NO lidar/ directory ---
```

**Drone LiDAR: confirmed absent in data, present in calib.** 0 `.ply` files on
the drone side in all four subsets, yet `lidar_bottom.json`,
`lidar_front.json`, `lidar_virtual.json` exist. Calib stubs without data. An
adapter must key modality off **file presence**, never off calib presence.

### Coordinate pipeline

```
pose/*.json  {x,y,z,roll,pitch,yaw,velocity,timestamp}   degrees, ENU world
      │  Rot.from_euler('xyz', [roll,pitch,yaw], degrees=True)
      ▼
   T_ego_to_ENU (4×4)          ← load_pose_griffin returns its INVERSE
      │
      ▼  AgentFrame.pose = inv(T_ENU_to_ego) = T_agent_to_world   ✓ canonical

lidar/lidar_top/*.ply  (N,4) x,y,z,I  in the LIDAR SENSOR frame
      │  calib/lidar_top.json extrinsic = T_lidar_to_ego
      │  = translate [0.25146, 0.0, 1.10431], rotation = I
      ▼  load_lidar(apply_mount=True)  →  EGO-frame points   ✓ canonical
```

Two conventions differ from OpenCOOD and must not be conflated:

| | OpenCOOD (OPV2V/V2XSet) | Griffin |
|---|---|---|
| pose vector | `[x,y,z,roll,yaw,pitch]`, CARLA `x_to_world` | `{x,y,z,roll,pitch,yaw}` dict, `scipy` Euler `'xyz'` |
| cloud frame on disk | agent/LiDAR frame (already) | **sensor** frame; mount applied by our loader |
| labels | one world-frame `vehicles` dict, shared | **per-agent** `.txt`, in that agent's own frame |
| world frame | CARLA map | ENU |

The per-agent label difference is structural: Griffin's drone has its own label
file with its own frame (drone-frame `z ≈ −26.7 m`, consistent with ~28 m
altitude). The canonical `Box3D.frame` field already expresses this
(`'agent'` vs `'world'`) — no structural change needed.

### Round-trip exactness — **it holds, for the same reason as OpenCOOD, but the claim is narrower**

OpenCOOD's round trip was bit-identical (`max|d| = 0.0`) because the pose is
converted **one way** (6-vector → 4×4) and `from_canonical` writes back a
*matrix* composed from matrices — the 6-vector is never reconstructed.

Griffin is the same shape: `pose.json → 4×4` one-way, and a Griffin
`from_canonical` would write matrices. So **the round trip is exact by the same
argument** — provided no Griffin consumer needs a 6-vector or Euler triple back.

**Where it would stop being exact, named:** `_pose6_from_matrix`-style inversion
(`arcsin`/`arctan2`) is *not* bit-exact and is gimbal-sensitive. The OpenCOOD
adapter only uses it under the non-default `write_back_pose=True`. If Griffin's
consumer turns out to need Euler back — which depends entirely on the info-pkl
schema in §5, currently unreadable — the round trip is **not** exact at that
site and the claim must be re-scoped. **I cannot verify Griffin's round trip
end-to-end without the pkl schema; claiming `max|d| = 0` now would be
unsupported.** What I can say: the *canonical-form* half is exact.

---

## 3. The injector set — **the premise is wrong in both directions**

### Finding A: V2XSet clouds are NOT zero-intensity; our own reader zeroes them

I was told fog/snow were excluded because they "silently no-op on zero-intensity
CARLA clouds". The observation was right; **the stated cause is wrong**, and the
real cause is a defect in our code.

Evidence:

```
V2XSet .pcd header:            FIELDS x y z rgb        ← no field named "intensity"
via OpenCOOD pcd_to_np():      I min 0.6196  max 0.9922  mean 0.9444  unique 96
via our src/datasets/pcd.py:   I min 0.0000  max 0.0000  mean 0.0000  unique 1
```

`load_pcd(path, columns=('x','y','z','intensity'))` documents "requested columns
absent from the file are filled with zeros". OpenCOOD stores intensity in the
**`rgb`** field (`pcd_to_np` reads `pcd.colors[:,0]`). No field is literally
named `intensity`, so our reader zero-fills — silently.

And fog is **not** inert on the real data. Same V2XSet cloud, read through
`pcd_to_np`, run through `LidarFogInjector`:

```
V2XSET sev1: in=39170 out=39170  meanI 0.9444->0.7979  xyz_moved= 9483  I_changed=39170
V2XSET sev2: in=39170 out=39170  meanI 0.9444->0.6559  xyz_moved=15089  I_changed=39170
V2XSET sev3: in=39170 out=39170  meanI 0.9444->0.5085  xyz_moved=28875  I_changed=39170
```

Backends and all three fog lookup tables are present (`MISSING_OPTIONAL == {}`).

**Consequences, none of which I am acting on:**
- The fog/snow exclusion for V2XSet rests on a false premise and is worth
  revisiting — **but that is the OpenCOOD path, explicitly out of scope here.**
- Fixing `src/datasets/pcd.py` touches shared code → **STOP-and-flag, per your
  rule.** It is a one-line field-alias fix (`intensity` ← `rgb`/`i`), but it
  changes what `src/datasets/opv2v.py` returns.
- **The OpenCOOD wrapper built in the previous phase is unaffected.** It takes
  `entry['lidar_np']` straight from OpenCOOD's `pcd_to_np` and never calls
  `load_pcd`. The verified baselines (V2X-ViT 0.84/0.62, CoBEVT 0.85/0.66,
  Where2comm) are not in question.

Note also: `opencood-official` is Python 3.7 and the fog lookup tables are
**pickle protocol 5** (3.8+). Fog raises `ValueError: unsupported pickle
protocol: 5` there. If fog is ever wanted on the OpenCOOD path it needs a
protocol downgrade or a different env — flagging so it is not discovered mid-run.

### Finding B: Griffin intensity is real — and irrelevant to the metric

Griffin `.ply` declares a literal `I` field and `load_lidar_ply` reads it
explicitly. Real, varying:

```
GRIFFIN cloud (2929,4):  I min 0.3175  max 0.9981  mean 0.9376  unique 877/2929
```

Fog demonstrably works:

```
GRIFFIN sev1: meanI 0.9376->0.7998  xyz_moved= 820  I_changed=2929/2929
GRIFFIN sev2: meanI 0.9376->0.7650  xyz_moved= 916  I_changed=2929/2929
GRIFFIN sev3: meanI 0.9376->1.0321  xyz_moved=1207  I_changed=2929/2929
```

Griffin's intensity distribution is essentially the same as V2XSet's
(mean 0.9376 vs 0.9444) — **both are CARLA. Intensity does not discriminate
between the two datasets.** The discriminator was our reader.

### But: the released cooperative model never reads LiDAR

From `logs/cooperative-train.log`, the dumped config:

```
model.pts_bbox_head = BEVFormerTrackHead
  transformer = PerceptionTransformer(num_cams=4)
    encoder = BEVFormerEncoder(SpatialCrossAttention(num_cams=4),
                               TemporalSelfAttention, MSDeformableAttention3D)
test_pipeline = [ LoadMultiViewImageFromFilesInCeph,
                  LoadInfInformation(keys=[query_feats, query_embeds,
                                           obj_idxes, ref_pts]),
                  NormalizeMultiviewImage, LoadAnnotations3D_E2E, ... ]
```

There is **no `LoadPointsFromFile`** in either pipeline. `num_cams=4` matches the
vehicle's four real cameras (front/back/left/right). The drone's five cameras are
**not** read at cooperative-eval time — the drone contributes *precomputed track
queries* loaded from disk.

### Resulting Griffin injector set

| injector | corrupts Griffin data? | moves the released cooperative metric? |
|---|---|---|
| **PoseError** | yes | **only if** the info-pkl seam is solved (§5) |
| **AgentDrop** (drone) | yes | **yes** — withhold the cached queries |
| **Camera injectors** (fog/darkness/brightness/occlusion/snow-image) | yes | **yes** — vehicle's 4 cams are the model input |
| **MissingModality-camera** | yes | yes |
| CommLatency | yes | probably — via query/frame staleness; unverifiable without the code |
| **LidarFog / LidarSnow** | **yes, verified above** | **NO** — model reads no LiDAR |
| **PointsReduce** | yes | **NO** — same reason |
| **MissingModality-LiDAR** | yes | **NO** — same reason |

**This is the inverse of the OpenCOOD set**, and it contradicts the expectation
stated in the task ("the 5 that worked on OpenCOOD PLUS fog/snow"). The honest
formulation: on Griffin, LiDAR injectors are **data-verified but metric-inert**;
the *measurable* set is the camera injectors plus PoseError plus AgentDrop.

(Caveat worth stating: `early-fusion` and `vehicle-side` checkpoints also exist
and I did not read their configs. If either has a LiDAR branch, LiDAR injectors
become measurable there. Cheap to check in Phase 1.5 if you want it.)

---

## 4. Mixed modality — the modality gate needs per-agent semantics

Griffin is the first dataset where agents differ in modality *within one sample*:
vehicle = LiDAR + 4 cameras; drone = 5 cameras, no LiDAR.

**The current gate is wrong for this.** `OpenCOODAdapter.assert_modality` raises
if **any** agent lacks the modality:

```python
missing = sorted(a for a, ag in sample.agents.items() if not _has(ag, need))
if missing: raise ModalityError(...)
```

On Griffin, `assert_modality(sample, 'lidar')` raises on **every** scene, because
the drone legitimately has none. The gate conflates "this dataset cannot support
this injector" with "this agent cannot receive it".

**Proposed fix — additive, OpenCOOD behaviour bit-identical.** Move the gate to a
shared `src/adapters/modality.py`:

- `agents_supporting(sample, need) -> [agent_id]` — the scope an injector runs on.
- `require_any(sample, need)` — raise only if **no** agent has it. This is the
  true "image injectors on LiDAR-only OpenCOOD data" block, and on V2XSet it
  raises exactly where the current code does, because *no* agent has images.
- OpenCOOD's adapter keeps calling a gate whose observable behaviour on
  LiDAR-only data is unchanged.

**This is still a change to `src/adapters/opencood.py`** (the static method moves)
→ flagged per your rule, not done. It is the minimum change I can find; the
alternative — a Griffin-local gate — duplicates the logic and lets the two drift.

### Injectors that would misbehave on a mixed scene

1. **MissingModality on the drone** — the drone's only modality is camera.
   Dropping it is *equivalent to dropping the agent*, but logs as a modality
   fault, so a results table would double-count it as a distinct failure mode.
   **Needs an explicit decision:** refuse (my recommendation — it is AgentDrop
   wearing a different label), or allow and log it as agent-equivalent.
2. **AgentDrop is not symmetric.** Dropping the vehicle removes the ego, all
   LiDAR, and the only model input — structurally forbidden. Dropping the drone
   removes the cooperative signal only. Griffin's ego is the *vehicle*, so
   `protect_ego=True` already covers this; worth asserting rather than assuming.
3. **Any `agent_scope='all'` sensor stage.** `FaultPipeline` applies
   `lidar_stages` to every agent whose `lidar is not None` — on Griffin the drone
   is skipped silently. Correct behaviour, but it means "PointsReduce, all
   agents" silently means "vehicle only". Must be logged, or a results table
   overstates coverage.
4. **PoseError is the one that is genuinely symmetric** — both agents have poses,
   both are perturbable, and the drone's is arguably the more interesting (a
   28 m-altitude platform with `pitch = −6.26°`, `roll = 1.31°`, i.e. real
   non-planar attitude, unlike CARLA ground vehicles). Note the standard
   planar-only protocol (`sigma_rollpitch = 0`) is a poorer fit for a drone;
   `sigma_rollpitch` is already a supported parameter.

---

## 5. Fire points — **the Griffin equivalent is worse than OpenCOOD's**

For OpenCOOD the problem was that `reform_param` baked the pose into
`transformation_matrix` *inside* `retrieve_base_data`, and the fix was to
recompute the matrix in `from_canonical`. There was a **load-time seam**.

For Griffin **there is no equivalent seam**, because the pose is consumed from a
**pre-generated info pickle**, baked at offline dataset-conversion time.

`CustomCollect3D` collects: `veh2inf_rt`, `l2g_r_mat`, `l2g_t`, `timestamp`,
plus meta keys `lidar2img`, `cam2img`, `can_bus`, `pcd_rotation`. In an mmdet3d
`AgileDataset`, these come from `get_data_info()` reading `agile_infos_*.pkl` —
not from `pose/*.json` at load time.

| injector | where it must fire on Griffin | no-op risk |
|---|---|---|
| **PoseError** | **the info-pkl record** (`veh2inf_rt`, `l2g_r_mat`, `l2g_t`, `lidar2img`, `can_bus`) — or regenerate the pkl from perturbed poses | Perturbing `pose/*.json` alone is a **silent no-op**, same class as `lidar_pose`. Worse: the fix is regenerate/patch the pkl, not recompute in `from_canonical`. |
| **Camera injectors** | on the loaded image array, i.e. between `LoadMultiViewImageFromFilesInCeph` and `NormalizeMultiviewImage` | Firing after normalisation/padding corrupts the wrong tensor; firing on disk is fine but re-reads. |
| **AgentDrop (drone)** | at `LoadInfInformation` — withhold/zero the cached query tensors | Removing drone *images* is a no-op: they are not read. |
| **CommLatency** | the query-file index in `LoadInfInformation`, and/or `timestamp`/`prev_idx`/`next_idx` | Griffin's tracker is temporal (`TemporalSelfAttention`, `queue_length`); a naive frame shift may corrupt tracking state rather than model latency. Unverifiable without the code. |
| **LiDAR injectors** | nowhere that matters | Model reads no LiDAR at all. |

**The structural finding:** OpenCOOD's fault plane sits at *dataset-load* time.
Griffin's sits partly at *dataset-conversion* time (offline, pkl) and partly at
*a cached intermediate produced by a different model* (track queries). A "thin
adapter at one seam" does not exist for Griffin the way it did for OpenCOOD.
**This is a design problem to resolve, not something to push through** — hence
this report rather than an implementation.

---

## 6. Does any of this require touching OpenCOOD / canonical / injectors?

**Qualified NO on the canonical form; YES on two shared items, both flagged.**

| item | verdict |
|---|---|
| `CooperativeSample` / `AgentFrame` | **No change.** Already per-agent `lidar`, `images`, `cameras`, `pose`, `labels` with `Box3D.frame`. Mixed modality is expressible today. |
| The five injectors | **No change** for PoseError/AgentDrop/CommLatency/MissingModality. |
| `src/adapters/opencood.py` | **CHANGE NEEDED** — `assert_modality` must gain per-agent semantics (§4). Proposed as a move to shared `src/adapters/modality.py` with OpenCOOD's observable behaviour unchanged. **Flagged, not done.** |
| `src/datasets/pcd.py` | **DEFECT FOUND** (§3 Finding A). Fixing it changes `src/datasets/opv2v.py` output. Does **not** affect the OpenCOOD wrapper or its baselines. **Flagged, not done.** |
| `PointsReductionInjector` | pre-existing global-reseed defect, already approved-to-fix, unrelated to Griffin. |

The verified OpenCOOD baselines are untouched by everything in this report.

---

## 7. What Phase 2 can and cannot be

The scope as written — "implement, gate, then PoC" — assumes a metric to degrade.
**There isn't one yet.** Three options; this is your call, not mine:

**A. Stand up the Griffin stack.** Clone the codebase, build a py3.8 /
torch 1.9.1+cu111 / mmdet3d env, run their converter to produce
`agile-nuscenes/` + info pkls, run drone-side inference to generate the
track-query cache, reproduce mAP 0.1442 / NDS 0.1972 as a control, *then* adapt.
Highest cost, only option that yields a real degradation curve. Note the env is
older than anything currently on this machine.

**B. Data-plane only.** Build the adapter, inject into Griffin samples, verify
and log corruption — no AP. Delivers a reusable Griffin adapter and auditable
fault data, but **cannot answer "how much does this degrade perception"**. Cheap,
and honest if labelled as such.

**C. Retarget to OpenCOOD.** §3 Finding A means fog/snow were excluded on a false
premise and demonstrably work on V2XSet clouds. Fixing `pcd.py` (or reading via
the wrapper's `lidar_np`, which already has intensity) would add two injectors to
a path that **already has verified baselines and a working wrapper**. Cheapest
path to a real degradation result — but it is OpenCOOD work, explicitly out of
this task's scope, and needs the py3.7 pickle-protocol issue resolved.

My recommendation if you want a measurable result soonest: **C**, then **A** when
there is time for the Griffin stack. **B** only if the Griffin adapter itself is
the deliverable.

**Phase 2 as specified is blocked on choosing among these.** I have not written
any adapter code.
