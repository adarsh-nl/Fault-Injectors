# Fault-injection wrapper — design report

**Status:** design only. Nothing built except the `plyfile` fix (§5), which was a
hard blocker on *reading* the design's prerequisites and is 3 lines.
**Awaiting approval before wiring.**

Scope: inject the 5 verified injectors (PoseError, CommLatency, AgentDrop,
MissingModality-LiDAR, PointsReduce) into the three verified baselines
(V2X-ViT/V2XSet, CoBEVT/V2XSet, Where2comm/OPV2V) through a **thin per-dataset
adapter onto the existing canonical `CooperativeSample`/`AgentFrame`**, with the
injectors themselves staying dataset- and framework-agnostic.

---

## 0. Architecture, and where the OpenCOOD-specific knowledge is confined

```
  OpenCOOD                     ADAPTER (per data format)        CANONICAL (agnostic)
  ────────                     ─────────────────────────        ────────────────────
  BaseDataset
    .time_delay_calculation ◄──── hook 1: pre-assembly ◄──── CommLatencyInjector
    .retrieve_base_data()
        │ base_data_dict
        ▼
      OrderedDict{cav_id: {ego, time_delay, params, lidar_np}}
        │                             to_canonical()
        └───────────────────────────────────────────────►  CooperativeSample
                                                              { AgentFrame(pose,
                                       hook 2: sample stage      lidar, is_ego) }
                                                                     │
                                    PoseError / AgentDrop / MissingModality  │
                                    / PointsReduce  (unchanged, agnostic)    │
                                                                     ▼
      OrderedDict (rebuilt) ◄──────── from_canonical() ◄────── CooperativeSample
        │
        ▼
  IntermediateFusionDataset.__getitem__  →  model  →  official eval  →  AP
```

Proposed files:

| file | imports | role |
|---|---|---|
| `src/adapters/opencood.py` | `numpy`, `src.datasets.base` only | `OpenCOODAdapter`: `to_canonical` / `from_canonical`. **No `opencood` import.** Pure dict-schema translation. |
| `src/adapters/runtime.py` | stdlib + `src.pipeline` | `make_faulty_dataset(base_cls, spec)` → a subclass with the two hooks. Takes the OpenCOOD class *as an argument*, so still no `opencood` import. |
| `tools/fi_inference.py` | `opencood` | ~40-line driver. The **only** file that knows OpenCOOD exists. |

Consequence for the layering rule: `src/` gains no new dependency, and Griffin/DAIR
later add `src/adapters/griffin.py` — a new `to_canonical`/`from_canonical` pair,
**no new injector and no rewrite**. That is the hard constraint, satisfied
structurally rather than by convention.

### Why a mixin over the official class, and not a fork of `inference.py`

`opencood/tools/inference.py` is the code path that produced 0.85/0.66 and
0.84/0.62. Forking it puts AP computation, NMS and result accumulation on a
branch that can silently drift. Instead the driver does:

```python
import opencood.data_utils.datasets as ocds
ocds.__all__['IntermediateFusionDataset'] = make_faulty_dataset(
    ocds.IntermediateFusionDataset, spec)
from opencood.tools.inference import main; main()          # unmodified
```

`build_dataset` reads `__all__` at call time, so this works. Everything from
`__getitem__` downward — preprocessing, model, `eval_final_results` — is the
byte-identical official code.

### One mixin covers both codebases

V2X-ViT was verified in `~/v2xvit-official`, a separate fork. I diffed
`v2xvit/data_utils/datasets/basedataset.py` against
`opencood/data_utils/datasets/basedataset.py`: **104 diff lines, all docstrings,
comments, and the `v2xvit.` ↔ `opencood.` namespace.** `retrieve_base_data`,
`reform_param`, `add_loc_noise` and `time_delay_calculation` are functionally
identical. Since `make_faulty_dataset` receives the base class as an argument and
the adapter imports nothing from either fork, the same code serves both.

---

## (a) OpenCOOD sample → `CooperativeSample` / `AgentFrame`

`retrieve_base_data(idx)` (`basedataset.py:199`) returns an `OrderedDict` keyed by
`cav_id`, ego first. Per entry:

| OpenCOOD key | canonical destination | notes |
|---|---|---|
| `cav_id` (dict key) | `AgentFrame.agent_id` | str |
| `['ego']` | `AgentFrame.is_ego`, `CooperativeSample.ego_id` | exactly one true |
| `['lidar_np']` | `AgentFrame.lidar` | `(N,4)` float32 x,y,z,intensity, **in the agent's LiDAR frame** — raw, pre-`shuffle_points`/`mask_ego_points`/range-mask. Exactly the canonical contract. |
| `['params']['lidar_pose']` | `AgentFrame.pose` via `x_to_world` | `[x,y,z,roll,yaw,pitch]` → 4×4 `T_agent_to_world` |
| `['params']['ego_speed']` | `AgentFrame.speed` | m/s (model input for V2X-ViT) |
| `['params']['vehicles']` | `AgentFrame.labels` (`Box3D`, `frame='world'`) | *optional*; see below |
| `['time_delay']` | `AgentFrame.faults['comm_latency']` / meta | int, units of 100 ms |
| `int(cav_id) < 0` | `AgentFrame.agent_type='infrastructure'` | V2XSet RSU convention |
| `['params']['transformation_matrix']`, `['gt_transformation_matrix']`, `['spatial_correction_matrix']` | **not carried into canonical form** | derived from poses; recomputed on the way back (§b) |

`AgentFrame.images` stays **empty** and `AgentFrame.cameras` stays **empty**: all
three baselines are LiDAR-only and `retrieve_base_data` never loads the PNGs.

**Modality gate (the structural block on image-injectors-on-LiDAR).** The check
lives in the canonical layer, once:

```python
def assert_modality(sample, need):        # need ∈ {'lidar','images'}
    missing = [a for a, ag in sample.agents.items()
               if not getattr(ag, 'images' if need == 'images' else 'lidar', None)]
```

`make_faulty_dataset` validates the spec against the *first* assembled sample and
raises at construction, not at frame 4000. Any image injector requested on a
V2XSet/OPV2V run dies before the model is built. Adding Griffin later flips the
same gate on automatically because its adapter populates `images`/`cameras` —
no new check, no per-injector `if`.

### Labels: carry them, but do not write them back

`params['vehicles']` is world-frame GT. It is used downstream **only** via
`generate_object_center(cav_contents, reference_lidar_pose=ego_lidar_pose)`
(`base_postprocessor.py:98`), which reads `params['vehicles']` and the **ego**
pose. I propose `to_canonical` populates `labels` (cheap, and Griffin/DAIR will
want it) but `from_canonical` passes `params['vehicles']` through **byref,
untouched**, with an assert that no stage mutated it. Rationale: GT must be
identical between clean and faulty runs or the AP delta is not attributable to
the fault.

---

## (b) Exactly where each injector fires, relative to OpenCOOD's transforms

### The load-bearing fact

`retrieve_base_data` → `reform_param` (`basedataset.py:402`) computes, **before
returning**:

```python
transformation_matrix    = x1_to_x2(delay_cav_lidar_pose, cur_ego_lidar_pose)
gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,   cur_ego_lidar_pose)
spatial_correction_matrix = eye(4)      # when cur_ego_pose_flag=True
```

and OpenCOOD's own `add_loc_noise` is applied to a **local copy** of the pose
(`delay_cav_lidar_pose`) — it never writes back into `params['lidar_pose']`.

Downstream, `get_item_single_car` consumes `params['transformation_matrix']`
(not `lidar_pose`) to project the cloud into the ego frame, and with
`proj_first=True` `get_pairwise_transformation` returns identity.

> **Therefore: perturbing `params['lidar_pose']` after `retrieve_base_data`
> returns is a silent no-op.** The matrices are already baked. This is the single
> thing that would make a naive wrapper report "PoseError has no effect."

**CORRECTED 2026-08-06.** This section previously claimed all three configs
use `fusion.args: []` and therefore `cur_ego_pose_flag=True`. That was
verified for CoBEVT and Where2comm only; **V2X-ViT ships
`cur_ego_pose_flag: False`**, which routes `reform_param` down its other
branch (`transformation_matrix` referenced to the *delayed* ego pose, and a
non-identity `spatial_correction_matrix` the model consumes). The claim was
written before V2X-ViT's checkpoint was located and was never re-validated
when it was.

| model | config | fusion args | `cur_ego_pose_flag` | wild_setting |
|---|---|---|---|---|
| CoBEVT | `~/opencood-eval/cobevt/config.yaml` | `args: []` | True (default) | `async: false`, `loc_err: false` (Perfect) |
| Where2comm | `~/opencood-eval/where2comm/config.yaml` | `args: []` | True (default) | `async: true`, `loc_err: true`, `xyz_std/ryp_std 0.2` (Noisy) |
| V2X-ViT | `/datasets/.../v2xset_checkpoints/v2x-vit/config.yaml` | `args: {cur_ego_pose_flag: False, ...}` | **False** | `async: true`, `loc_err: true` (Noisy) |

**Why it no longer matters.** The delta composition (§ below) writes
`T_new = T_orig @ inv(A_clean) @ A_perturbed`. The ego reference `E` and the
baked agent pose `A_b` appear **only inside `T_orig`**, which is taken
verbatim from OpenCOOD, so the form is invariant to which branch
`reform_param` took. `spatial_correction_matrix` is never touched.

Verified empirically on V2X-ViT (`cur_ego_pose_flag=False`, `loc_err=true`),
220 poses: null-spec and zero-sigma both give `max|dT| = 0.0000` and
`max|d spatial_correction| = 0.0e+00`; sigma=0.4 gives `max|dT| = 1.1389`
with the correction matrix still untouched. The old recompute-from-clean-pose
form was **not** invariant and produced `max|dT| = 0.205` with no fault
injected.

### SUPERSEDED 2026-08-06: recompute was wrong; compose a delta instead

The recompute described immediately below was the original resolution and it
**silently discarded OpenCOOD's own `add_loc_noise` perturbation**, which is
applied to a local copy inside `reform_param` and never written into
`params['lidar_pose']`. Rebuilding the matrix from that clean pose therefore
stripped the shipped noise for any model with `loc_err: true`. Measured with
NO fault injected: `max|dT|` = 0.177 (Where2comm), 0.205 (V2X-ViT), 0.000
(CoBEVT, which ships Perfect and so was unaffected).

**The correct form composes the perturbation as a right-multiplied delta onto
the matrix OpenCOOD baked:**

```
T_new = T_orig @ inv(A_clean) @ A_perturbed
```

derived from the convention read at source: `x_to_world` = `T_x->world`
(point-transform), `x1_to_x2(cav, ego)` = `T_agent->ego`, and the consumer
`project_points_by_matrix_torch` computes `p_ego = T @ p_agent`. With
`T_orig = inv(E) @ A_b` and a broadcast pose `A' = A @ D`, we get
`inv(E) @ (A_b @ D) = T_orig @ D` — right-multiply. Left-multiplying would
apply the delta in the *ego* frame (a rotation about the ego origin):
plausible magnitudes, wrong geometry, and it would pass a magnitude-only
fire-check.

When the pose is untouched the matrix is left byte-identical rather than
multiplied by a near-identity, so the reduction is exact (measured 0.0 over
220 poses, both null-spec and sigma=0).

The original text follows for the record.

### (superseded) recompute the matrices in `from_canonical`

`from_canonical` rebuilds `transformation_matrix` from the (possibly perturbed)
canonical poses:

```
T = inv(ego.pose) @ agent.pose
```

which is *definitionally* `x1_to_x2(cav_pose, ego_pose)` = `inv(x_to_world(ego)) @
x_to_world(cav)`. This is also exactly what `CooperativeSample.lidar_in_ego_frame`
already computes (`src/datasets/base.py:117`) — the canonical model already
encodes the quantity pose error corrupts.

**Verified prerequisite:** our `x_to_world` (`src/datasets/opv2v.py:47`) vs
OpenCOOD's (`transformation_utils.py`), 200 random poses over ±200 m / ±180°:

```
max abs diff x_to_world: 0.0
```

Bit-identical. So the null-pipeline round trip is exact, not approximate.

### Per-injector firing points

| injector | plane | fires | frame |
|---|---|---|---|
| **PoseError** | corruption | on `AgentFrame.pose` in the sample stage; `from_canonical` then **recomputes** `transformation_matrix`. Effectively pre-transform. | world (perturbs `T_agent_to_world`) |
| **CommLatency** | corruption | **hook 1**, `time_delay_calculation` — *pre-assembly* | frame index |
| **AgentDrop** | corruption | sample stage; agent removed from `sample.agents`, then from the rebuilt `OrderedDict` | n/a |
| **MissingModality-LiDAR** | corruption | sample stage (needs a thin wrapper, below) | agent frame |
| **PointsReduce** | corruption | sensor stage on `AgentFrame.lidar` — the **raw** cloud, before `shuffle_points` / `mask_ego_points` / `mask_points_by_range` | agent (LiDAR) frame |

**PoseError — the two things that decide whether the PoC is valid.**

1. *Ego must never be perturbed.* `PoseErrorInjector.apply_to_sample` defaults
   `protect_ego=True`, but `FaultPipeline.apply_to_sample` calls `fn(sample)`
   with no kwargs (`src/pipeline.py:98`), so it takes the default. I verified
   this path end-to-end rather than assuming it:

   ```
   ego pose unchanged: True
   c1  pose changed  : True
   c1 fault log      : {'pose_error': {'dx': 0.0691, 'dy': 0.1643, 'dz': 0.0,
                                       'dyaw': 0.0661, 'droll': 0.0, 'dpitch': 0.0}}
   ```

   If ego pose ever moved, `reference_lidar_pose` moves with it, GT moves, and AP
   drops for a reason that has nothing to do with cooperative misalignment — a
   "PoseError works!" result that is actually a broken-GT result.

2. *`params['lidar_pose']` stays clean.* For non-ego agents it reaches exactly
   one consumer after our seam: the 70 m `COM_RANGE` gate in
   `intermediate_fusion_dataset.py`. Leaving it clean is what OpenCOOD's own
   `add_loc_noise` does, so it is protocol-faithful; and at σ=0.2 m it is
   behaviourally a no-op except at measure zero. Recording this so it does not
   get re-litigated: **not a semantic choice, a no-op that happens to match the
   published protocol.** The noisy pose is written to a separate logging key.

**CommLatency — the one genuine seam mismatch, stated rather than smoothed over.**
Every other injector runs on the assembled `CooperativeSample`. CommLatency
cannot: it decides *which file to read*, upstream of assembly. So the mixin has
**two hook points, not one**. Overriding `time_delay_calculation` is the native
seam and preserves `time_delay` as a V2X-ViT model input feature (it is consumed
by the prior-encoding branch). Unit conversion, stated explicitly because it is
easy to get wrong: OpenCOOD's `time_delay_calculation` returns an **integer in
units of 100 ms** (`time_delay = time_delay // 100`) and is then clamped against
`timestamp_index`; our `CommLatencyInjector.stale_index` is **frame-based** with a
`k_min` floor. At 10 Hz, 1 frame = 1 unit. The adapter maps frame-shift ↔ integer
delay; nobody passes seconds.

Under async, `delay_params['lidar_pose']` is the *delayed* cav pose while
`gt_transformation_matrix` needs the *current* one, which is not in the returned
dict — so it cannot be faithfully recomputed from our seam. Resolution: the
source comment says `gt_transformation_matrix` "is only used for late fusion",
and all three models are intermediate fusion, so it is unused. `from_canonical`
will **assert it is unused** rather than reconstruct it. If a late-fusion model
is ever added, that assert fires and says so. Same for
`spatial_correction_matrix`: identity under `cur_ego_pose_flag=True`, which all
three configs use.

**PointsReduce — frame is right, but two things need saying.**
It fires on the raw agent-frame cloud, which is the sensor plane and matches
MultiCorrupt's semantics. Consequence to state out loud: sev 2 keeps 20 % of the
**raw** cloud, so the post-range-filter count is *not* 20 % of the clean run's
post-filter count. Anyone comparing point counts across conditions needs that.

*Defect to fix before use:* `PointsReductionInjector.__call__`
(`src/fault_injectors/lidar_points_reduce.py:18`) does `np.random.seed(self.seed)`
with a **fixed** seed, on the **global** numpy RNG, on **every call**. That means
(a) every agent and every frame gets the identical permutation prefix, and (b) it
stomps the global RNG that `time_delay_calculation`'s `np.random.uniform` and
`add_loc_noise` also draw from. For a per-sample-seeded, logged design this is a
real bug. Fix: per-`(idx, agent_id)` derived seed and a local `Generator`, no
global reseed. This is a change to an injector, so I am flagging it rather than
doing it silently.

**MissingModality-LiDAR** has no `apply_to_sample` — only `inject(image, points)`
(`missing_modality.py:142`). Wiring `drop_points` as a `lidar_stages` callable
would lose the per-agent gate semantics ("this agent's LiDAR is dead"). Proposal:
a ~10-line sample-level adapter in `src/adapters/` that calls `bernoulli_mask`
per agent and applies `drop_points`, logging `m_lidar` to `agent.faults`. The
injector itself is untouched. `p_drop_rgb` stays hard-asserted to 0.

I tested the downstream consequence rather than leaving it as an unknown —
`SpVoxelPreprocessor` on a 0-row cloud, under the real CoBEVT config:

```
0 rows -> {'voxel_features': (0, 32, 4), 'voxel_coords': (0, 3), 'voxel_num_points': (0,)}
```

No crash; it yields zero voxels. **Still unknown:** whether `collate_batch_test`
and the PointPillars scatter tolerate one agent contributing zero voxels. Listed
in §7 as the one remaining runtime unknown.

---

## (c) Seeded and logged per-sample injection

**The constraint that drives the whole design:** `opencood/tools/inference.py:62`
uses `num_workers=16`. `retrieve_base_data(idx)` therefore runs in 16 forked
processes. Any injector RNG state living on the dataset object is **per-worker**,
and the stateful injectors (`AgentDropInjector` carries Gilbert-Elliott
`_bad_state`; `PoseErrorInjector` holds one `default_rng`) would not reproduce
across worker counts — a run at `num_workers=16` and the same run at 8 would give
different numbers.

So: **stateless per sample.** Every draw derives from the frame index and agent
id, never from accumulated state:

```python
seq = np.random.SeedSequence(entropy=base_seed,
                             spawn_key=(idx, stable_hash(agent_id), stage_ord))
inj = PoseErrorInjector(sigma_xy=..., seed=seq.generate_state(1)[0])
```

Injectors are constructed **per call** inside `retrieve_base_data`, not held on
`self`. Cost is negligible next to the pcd read. Properties this buys:

- identical results at any `num_workers`, including 0;
- every row of the log is independently re-derivable from `(base_seed, idx, agent_id)`;
- `shuffle=False` in the eval loader, so `idx` is a stable key.

**Caveat to accept, not hide:** Gilbert-Elliott burst drop is inherently
sequential — a stateless-per-`idx` formulation cannot reproduce a Markov chain
across frames. For bursty AgentDrop I propose deriving the chain from
`SeedSequence(base_seed, spawn_key=(scenario_id, agent_id))` and advancing it
`timestamp_index` steps, which is reproducible but O(k) per sample. i.i.d.
AgentDrop (what the PoC and most conditions use) has no such issue. Flagging it
now so it is not discovered later.

**Hook 1 does not receive `idx` or `cav_id`.** `time_delay_calculation(self,
ego_flag)` takes only `ego_flag`, so at that hook neither key needed for the
derivation above is in scope; its call site is one frame up, inside
`retrieve_base_data`'s `for cav_id, cav_content in scenario_database.items()`
loop. Resolution, stated so "stateless per sample" and "stash `idx` on `self`" do
not read as a contradiction: the `retrieve_base_data` override (which we are
writing anyway for hook 2) computes the **whole per-agent delay map up front**
from `(idx, cav_ids)`, and the `time_delay_calculation` override is a pure
lookup into that map. No per-call mutable state, no ordering dependence.

**`wild_setting` policy for the two Noisy baselines — a design decision, not an
implementation detail.** Per the config table in §b, the *verified* Where2comm
and V2X-ViT baselines run with `loc_err: true, xyz_std: 0.2, ryp_std: 0.2` and
`async: true`. Injecting our faults on top of that stacks two noise sources and
the AP delta stops being attributable — and `add_loc_noise`'s fixed global
reseed means OpenCOOD's contribution is the *same draw* for every agent and every
frame, so it is not even a well-behaved nuisance term. Therefore: **when our
injectors run, `wild_setting.loc_err` and `wild_setting.async` are forced
`false`**, and the comparison baseline for those two models is re-measured clean
at that setting. Irrelevant to the CoBEVT PoC (Perfect — both already false), but
it bites on the very next model.

**Logging.** `agent.faults` is already the canonical channel —
`PoseErrorInjector` writes `agent.faults['pose_error'] = {dx, dy, dz, dyaw, ...}`
(verified above), `AgentDropInjector` writes `sample.meta['dropped_agents']`.
`from_canonical` drains those into one row per (idx, agent, stage), appended to
`injection_summary.csv` in the repo's existing results-bundle shape
(§"Results bundle" in `CLAUDE.md`), with the resolved seed alongside the drawn
values. Writes go through a per-worker file (`injection_summary.<pid>.csv`)
concatenated at the end — 16 processes appending to one CSV would interleave.

---

## (d) The `plyfile` blocker — resolved

**Diagnosis.** `src/datasets/__init__.py:33` eagerly imports `griffin`, which
imports `src/data_loaders.py`, which had `from plyfile import PlyData` at module
level. `plyfile` is absent from `opencood-official`. Because importing *any*
submodule runs the package `__init__`, even `import src.datasets.base` failed:

```
FAIL src.datasets.base | ModuleNotFoundError No module named 'plyfile'
```

I checked what else that chain needs: `PIL`, `scipy`, `numpy`, `yaml`, `einops`
are all present in `opencood-official`. `plyfile` was the only gap.

**Rejected: `pip install plyfile` into `opencood-official`.** That is the env
which produced the verified 0.85/0.66 and 0.84/0.62 baselines (Python 3.7.16,
numpy 1.21.6, torch 1.12.1+cu113). `~/CLAUDE.md` forbids bare installs; a
resolver bump to numpy or scipy would make every subsequent run non-comparable to
the baseline it is being compared against. A code fix touches nothing that
produced a number.

**Applied: lazy import.** `plyfile` is used at exactly one site
(`data_loaders.py:83`, `load_lidar_ply`) and is Griffin-only — OPV2V/V2XSet ship
`.pcd`, DAIR-V2X ships `.pcd`/`.bin`. Moved the import inside the function, with
a comment explaining why. This matches the repo's existing convention (lazy
`torchvision` so LiDAR-only runs don't need it; "optional backends degrade rather
than break").

**Verified after patching, in the target env** (not assumed):

```
src.datasets imports OK: ['dair-v2x', 'griffin', 'opv2v', 'v2xset']
```

All 5 injector modules and `src.pipeline` also import clean under Python 3.7.16.

This is the only code I have changed. Revert = move one line back.

---

## 6. Verification gates — what must pass before any number is believed

**Gate 1 — null-pipeline field-level round trip.** With an empty pipeline,
`from_canonical(to_canonical(d))` must reproduce `d`:

- `transformation_matrix` — exact (`x_to_world` is bit-identical);
- `lidar_np` — array equality, **dtype, and column count**. `pcd_to_np` gives
  `(N,4)`; `shuffle_points`/`mask_ego_points` run *after* our seam, so the shape
  contract must be handed back intact;
- **`OrderedDict` key order, ego first.** `intermediate_fusion_dataset.py`
  hard-asserts `cav_id == list(base_data_dict.keys())[0]` for ego. AgentDrop
  rebuilds this dict; if ego-first ordering is lost you get an `AssertionError`,
  not a silent wrong number. (Python 3.7 dicts preserve insertion order and
  deletion preserves the rest, so `AgentDropInjector.apply_to_sample`'s
  `del sample.agents[a]` is safe — but it must be asserted, not assumed.)
- `params['vehicles']`, `params['ego_speed']`, `gt_transformation_matrix`,
  `spatial_correction_matrix`, `time_delay`, `ego` — passed through untouched,
  asserted so a field cannot quietly get dropped in `from_canonical`.

**Gate 2 — null-pipeline AP, against a same-session control.**

The obvious formulation — "must reproduce exactly 0.85 / 0.66" — is **not a
usable gate**, and it is worth writing down why before someone tries it and
chases a phantom. `get_item_single_car` calls `shuffle_points(lidar_np)`, which
is `np.random.permutation` on the **global** RNG (`pcd_utils.py:92`). Point order
decides which points survive per-pillar truncation — `max_points_per_voxel: 32`
in the CoBEVT config — so features, and AP at full precision, depend on that RNG.
Nothing in the eval path seeds it, so each worker's state at fork is OS entropy:
different every run. And 0.85 / 0.66 is a 2-d.p. number remembered from a run
whose worker count and RNG state were not recorded, while `eval_final_results`
prints more digits.

Fix: **`tools/fi_inference.py` seeds the global numpy RNG before calling
`main()`.** With `shuffle=False` and deterministic worker-index assignment, that
pins the path. The gate then becomes legitimate rather than abandoned:

- run wrapper-null **twice** → bit-identical AP. If not, something else is
  stochastic (check `DataAugmentor.__init__(params, train=False)`);
- **Gate 2 proper: wrapper-null AP == a fresh *official* `inference.py` run in
  the same session, same seed, same `num_workers`, compared at full printed
  precision.** Against a run just performed, not against a remembered 0.85/0.66.
  (0.85/0.66 remains the sanity check that the control is the right model.)

This is what makes the sev-2 delta attributable: `PoseErrorInjector` uses a local
`np.random.default_rng`, so it does **not** disturb the global RNG that
`shuffle_points` draws from. `PointsReductionInjector` is the exception — see the
global-reseed defect in §b — and OpenCOOD's own `add_loc_noise` has the identical
defect (`np.random.seed(self.seed)`, fixed, global, every call), which is a
second reason for the `wild_setting` policy below.

**Gate 3 — the fault actually fires.** `injection_summary.csv` non-empty, one row
per non-ego agent per frame, and — concretely, so a factor-of-2 unit slip cannot
pass — mean `|dx|` ≈ σ√(2/π) = **0.160 m** at σ = 0.2.

**Gate 4 — ego untouched.** Assert ego pose identical between clean and faulty
runs, per sample.

Only after 2 and 4 pass does a sev-2 AP drop mean anything.

---

## 7. Open risks

1. **Zero-voxel agent.** Preprocessor survives (measured); `collate_batch_test` +
   PointPillars scatter with one agent at zero voxels is untested. Affects
   MissingModality-LiDAR only, not the PoseError PoC.
2. **`PointsReductionInjector` global reseed** (§b) — must be fixed before
   PointsReduce is used; it also perturbs OpenCOOD's own RNG.
3. **Bursty AgentDrop reproducibility** under `num_workers>0` (§c). i.i.d. is fine.
4. **`eval_final_results` writes into `opt.model_dir`.** Per-condition runs would
   overwrite each other; the driver needs a per-condition output dir or to capture
   `result_stat` directly.

Not a risk, worth stating: **AgentDrop rides an already-exercised path.** The
`COM_RANGE` `continue` in `__getitem__` (`distance > 70` → skip the cav) already
produces variable agent counts every epoch, so a removed agent is a condition the
collate and the model handle routinely — it is not a new code path.

---

## 8. PoC plan (on approval)

1. Build `src/adapters/opencood.py` + `runtime.py` + `tools/fi_inference.py`
   (driver seeds global numpy).
2. Gate 1 (unit, login node, seconds).
3. Control run: **official, unmodified** `inference.py` on CoBEVT / V2XSet with
   the seed and worker count fixed → record AP at full precision.
4. Gate 2: same job, wrapper with null pipeline → must match step 3 at full
   precision, and be bit-identical across two repeats.
5. PoseError sev 2 (σ_xy = 0.2 m, σ_yaw = 0.2°, non-ego only) → expect AP below
   the step-3 control, with Gates 3 (mean |dx| ≈ 0.160 m) and 4 green.

sbatch, `--constraint=a40|a100|l40|l40s`, shown before submission per `~/CLAUDE.md`.
No git, no sweep.

---

## 9. Implementation record (built, approved design)

Approved decision on flag 4: **`wild_setting.loc_err` and `.async` are forced
`false` whenever our injectors run**, and the clean control is measured at that
same forced setting. `tools/fi_inference.py` patches `load_yaml`, asserts both
are false, and prints what they were. No-op for CoBEVT (Perfect).

Files:

| file | what |
|---|---|
| `src/adapters/opencood.py` | `OpenCOODAdapter.to_canonical` / `.from_canonical` / `.assert_modality`. No OpenCOOD import. |
| `src/adapters/runtime.py` | `FaultSpec`, `make_faulty_dataset(base_cls, spec)`, `_seed`, per-worker `_InjectionLog`. No OpenCOOD import. |
| `tools/fi_inference.py` | the only OpenCOOD-importing file; seeds numpy, swaps `__all__`, calls official `main()` unmodified. |
| `tools/test_gate1.py` | Gate 1, 47 checks, login-node. |
| `tools/smoke_adapter.py` | CPU-only smoke on real V2XSet. |
| `tools/slurm/fi_poc_cobevt.sbatch` | the four-pass PoC job. |

### Deviations from §a–§c, and why

1. **`load_labels` defaults `False`** (§a said populate). No injector in the
   verified set touches labels, `params['vehicles']` is passed back by reference
   untouched, and building ~50 `Box3D` per agent per frame across 16 workers is
   pure overhead. The flag exists and Griffin will set it.
2. **PointsReduce raises `NotImplementedError`** rather than running. The
   global-reseed defect is approved-to-fix but not yet fixed; refusing beats
   emitting a wrong number.
3. **Bursty AgentDrop is not implemented** — i.i.d. only, as designed. The
   Gilbert-Elliott path is unreachable from `FaultSpec`.
4. **Hook 1 (CommLatency) is wired but unexercised.** With `spec.latency=None`
   it delegates straight to OpenCOOD's own method, so the PoseError PoC does not
   touch it. It gets its own PoC when CommLatency lands.

### Results of the login-node gates

- **Gate 1** — 47/47 pass. `transformation_matrix` round-trip
  `max|d| = 0.000e+00` (bit-identical). Key order, ego-first, lidar dtype/shape/
  columns, all five pass-through fields, modality gate, drop-keeps-ego-first,
  ego-drop refused, seed stability/variation.
- **Real-data smoke** (`smoke_adapter.py`, V2XSet `2021_08_18_19_48_05`,
  178 samples, real `reform_param` + `pcd_to_np`) — 0 failures. Null wrapper
  reproduces the official dict exactly; sev-2 moves only non-ego
  `transformation_matrix`; ego untouched; `gt_transformation_matrix` untouched;
  **GT boxes identical clean vs faulty**; `__getitem__` survives the rebuilt
  dict through to preprocessed features on both paths.
- **Gate 3 estimator** — 4000 draws through the stateless
  `SeedSequence(spawn_key=(idx, crc32(agent), stage))` path:
  `mean|dx| = 0.1592` (expected `σ√(2/π) = 0.1596`), `std = 0.2002`,
  `std dyaw = 0.2031°`. No seed-correlation artifact.

Gate 2 (wrapper-null == fresh official control, at full precision) and Gate 4
require the GPU job.

## 10. pcd intensity fix + fog/snow re-enable (2026-08-05)

- `src/datasets/pcd.py`: a requested `intensity` column now falls back to an
  `I`/`i` field, then to unpacking the PCL bit-packed `rgb` float (red byte,
  `((bits>>16)&0xFF)/255`) — bit-identical to OpenCOOD's `pcd_to_np`
  (max|d| = 0.0 over 10 V2XSet files, all 4 columns). Before this, the column
  was silently zero-filled, which is the *actual* reason LidarFog/LidarSnow
  looked like no-ops on OPV2V/V2XSet.
- Baseline isolation: official OpenCOOD repo has zero references to this
  codebase; the wrapper consumes `lidar_np` from `pcd_to_np`; `load_pcd`'s
  only callers are `src/datasets/{opv2v,dair_v2x}.py`. Post-fix confirmation
  job: `tools/slurm/fi_baseline_cobevt.sbatch`.
- Fog tables re-dumped pickle protocol 4 as pure-Python floats (value-exact;
  originals recoverable via git) — they were protocol 5 with numpy scalars,
  unreadable in the py3.7 wrapper env. Fog output py3.7 vs py3.13:
  bit-identical on a full V2XSet cloud.
- `LidarSnowInjector.__call__` now save/restores the global numpy RNG around
  the verbatim sim (same defect class PointsReduce is blocked on; output
  verified unchanged, caller's stream verified preserved). Snow output is
  env-dependent (numpy 1.21 vs 2.5 resolve the RANSAC noise-floor fit
  differently) but deterministic within the wrapper's env.
- Modality gate moved to `src/adapters/modality.py` (`require_all` /
  `require_any` / `agents_supporting`); `OpenCOODAdapter.assert_modality`
  delegates with a dataset hint — observable behaviour byte-identical across
  five captured cases including edge cases.
- `FaultSpec` gained `lidar_fog` / `lidar_snow` (environmental → default
  scope `'all'`, ego included; per-(idx,agent,stage) seeding; snow particle
  files pre-generated in the parent process to avoid a 16-worker write race).
  Wrapper-level fog fires-test on real V2XSet: every agent's cloud changed,
  transformation matrices untouched, `__getitem__` clean.
- OpenCOOD injector set is now 7: PoseError, CommLatency, AgentDrop,
  MissingModality-LiDAR, PointsReduce (blocked in wrapper pending its own
  approved RNG fix), LidarFog, LidarSnow. For V2XSet snow pass
  `mount_height=1.9`.

### Expected magnitude — recorded BEFORE the run, so the result can falsify it

σ_xy = 0.2 m / σ_yaw = 0.2° is the **mildest** point on the standard V2X-ViT
robustness sweep. The expected signature is a drop of a few tenths of a point at
AP@0.7 and near-nothing at AP@0.5 — **not** a collapse.

So: if `pose_sev2` lands within the determinism floor measured by
`clean` vs `clean_rep`, that is **not** evidence the injection failed. Injection
is already proven independently — non-ego `transformation_matrix` moves, ego's
does not, and mean `|dx| = 0.1592` against the theoretical `0.1596`. It would
mean sev 2 sits below this model's noise floor on this split, which is itself a
finding, and the next step is σ = 0.4–0.5, **not** debugging the adapter.

Read `clean` vs `clean_rep` first when results land: it says whether the
full-precision Gate 2 comparison is meaningful at all.
