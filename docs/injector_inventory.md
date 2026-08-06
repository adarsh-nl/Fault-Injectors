# Fault-injector inventory

Complete audit of `src/fault_injectors/` (2026-08-06), read from registry,
constructors and docstrings — not inferred from names. "Sweep" = the
7-injector OpenCOOD grid (seed 1234, per-sample stateless SeedSequence);
"Griffin wrapper" = the gate-verified `GriffinFaultSpec` routing (unmeasured
until a Griffin model exists); "viz" = `notebooks/fault_visualisation.py`
exports unless marked old-ipynb-only.

| # | Class | File | Modality | Severity parameter (range / tested tiers) | Valid datasets | Wired into | Excluded from OpenCOOD sweep because |
|---|---|---|---|---|---|---|---|
| 1 | `PoseErrorInjector` | `pose_error.py` | cooperative (shared poses) | `sigma_xy` m, `sigma_heading` deg, >=0 (+ optional `sigma_z`, `sigma_rollpitch`; gaussian\|laplace) / 0.2, 0.4, 0.6 | OpenCOOD yes; Griffin NO (not wired by approved routing) | sweep + viz | — (in sweep) |
| 2 | `CommLatencyInjector` | `communication.py` | cooperative (frame staleness) | `mu_delay` s >=0, `sigma_jitter` s >=0 / 0.1, 0.2, 0.3 s | both (Griffin scene-clamped, drone) | sweep + viz | — (in sweep) |
| 3 | `AgentDropInjector` | `communication.py` | cooperative (transmission loss) | `p_drop` in [0,1] i.i.d.; optional Gilbert-Elliott `burst{p_bad,p_recover}` (burst not sweep-wired) / 0.25, 0.50, 0.75 | both (Griffin: drone only) | sweep + viz | — (in sweep) |
| 4 | `MissingModalityInjector` | `missing_modality.py` | sensor dropout (lidar or camera) | `p_drop_lidar`, `p_drop_rgb` in [0,1] / 0.25, 0.50, 0.75 | both (OpenCOOD: lidar gate; Griffin: drone camera gate) | sweep + viz | — (in sweep; `p_drop_rgb` hard-asserted 0 on OpenCOOD) |
| 5 | `PointsReductionInjector` | `lidar_points_reduce.py` | lidar | `severity` in {1,2,3} -> keep 30/20/10 % | both | sweep + viz | — (in sweep; global-RNG defect fixed 2026-08-05) |
| 6 | `LidarFogInjector` | `lidar_fog.py` | lidar | `severity` in {1,2,3} (visibility 300/150/50 m); optional `T_lidar_to_ego` | both (real intensity confirmed on both after the pcd `rgb`-unpack fix) | sweep + viz | — (in sweep; early exclusion traced to our zero-filling reader, not the data) |
| 7 | `LidarSnowInjector` | `lidar_snow.py` | lidar | `severity` in {1,2,3} (5/35/70 mm/h); `mount_height` (1.9 CARLA / 1.10 Griffin), `T_lidar_to_ego` | both (Griffin: severity-flat removal caveat) | sweep + viz | — (in sweep) |
| 8 | `BandwidthLimitInjector` | `communication.py` | cooperative (partial point sharing) | `keep_fraction` in (0,1], optional `quantise_m` | both in principle | neither (FaultPipeline-reachable only) | dropped during injector verification: found mislabeled as a bandwidth model — semantically overlaps PointsReduce on the transmitted cloud |
| 9 | `TemporalMisalignmentInjector` | `temporal_misalignment.py` | cross-modal (stale image + current lidar, one agent) | `mu_delay` s (default 0.2), `sigma_jitter` s (default 0.05) | Griffin only (needs images+lidar on one agent) | old `.ipynb` viz only — not yet in the new marimo exports | cross-modal by construction; OpenCOOD pipeline is LiDAR-only (no image to misalign) |
| 10 | `CalibrationErrorInjector` | `calibration.py` | camera geometry (K + extrinsic drift) | `sigma_focal_px`, `sigma_principal_px`, `sigma_translation_m`, `sigma_rotation_deg`, all >=0 | Griffin only (camera calib) | neither | camera-geometry fault on a LiDAR-only pipeline; also not wired into the Griffin spec yet |
| 11 | `SensorOcclusionInjector` | `sensor_occlusion.py` | camera surface (dirt/scratch/crack) | `OcclusionConfig`: `kind` in {dirt,scratch,crack}, `severity` float (opacity OR coverage mode, continuous), placement random\|targeted\|avoid, temporal iid\|persistent | Griffin only | viz (old + new) | camera fault; modality gate blocks on LiDAR-only OpenCOOD |
| 12 | `BrightnessInjector` | `brightness.py` | camera | `severity` in {1,2,3} (HSV V +0.5/0.6/0.7); 4/5 raise | Griffin only | Griffin wrapper (`camera.kind`) + viz | camera fault on LiDAR-only pipeline (structural gate) |
| 13 | `DarknessInjector` | `darkness.py` | camera | `severity` in {1,2,3} (noise-s 25/12/5, falls as fault worsens) | Griffin only | Griffin wrapper + viz | same as #12 |
| 14 | `FogInjector` (camera) | `fog.py` | camera | `severity` in {1,2,3} (visibility 300/150/50 m) | Griffin only | Griffin wrapper + viz | same as #12 |
| 15 | `SnowInjector` (camera) | `snow.py` | camera | `severity` in {1,2,3} (5/35/70 mm/h) | Griffin only | Griffin wrapper + viz | same as #12 |
| 16 | `BeamReductionInjector` | `lidar_beams_reduce.py` | lidar (scan lines) | `severity` in {1,2,3} -> keep 16/8/4 of 32 beams; REQUIRES (N,5) ring column | neither current dataset (nuScenes-style only; raises on (N,4)) | neither | CARLA clouds carry no ring index — raises rather than misreading intensity as ring |
| 17 | `BeamReductionInjectorGriffin` | `lidar_beams_reduce.py` | lidar (elevation-binned) | `severity` in {1,2,3} -> keep 1/2, 1/4, 1/8 of `num_beams` (80) | Griffin only (ring-free derivation) | viz (old + new) | Griffin-native variant; ring-free binning defined for Griffin's 80-beam geometry, not wired to the sweep grid |

**Footnote — `motion_blur`:** MultiCorrupt backend present in `_mc_image.py`
with the correct level-to-sigma_t mapping (0.06/0.10/0.13), but no wrapper
class exists; recorded in `severity.py`'s `UNAVAILABLE` and excluded from
`make()` rather than being silently absent.

Known gap surfaced by this audit: `TemporalMisalignmentInjector` is the one
injector with no coverage in the new visualization exports (old notebook
only).
