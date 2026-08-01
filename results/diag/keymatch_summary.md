# Keymatch summary

| model | verdict | exact name+shape | rename-only upper bound | official tensors | reimpl tensors |
|---|---|---:|---:|---:|---:|
| [cobevt](cobevt_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 69.14% | 243 | 195 |
| [v2xvit](v2xvit_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 82.24% | 304 | 281 |
| [where2comm](where2comm_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 82.43% | 148 | 126 |
| [cora](cora_keymatch.md) | `NO-CHECKPOINT` | -% | -% | - | - |
| [cobevt_camera_dynamic](cobevt_camera_dynamic_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 96.76% | 648 | 627 |
| [cobevt_camera_static](cobevt_camera_static_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 97.06% | 646 | 627 |
| [cobevt_lidar_opv2v_nocomp](cobevt_lidar_opv2v_nocomp_keymatch.md) | `NOT-WORTH-IT` | 0.0% | 75.68% | 222 | 195 |

**Verdict rule.** On `pct_official_exact` = share of official tensors matched
by *both* name and shape:

| verdict | rule | meaning |
|---|---|---|
| `CHEAP` | >= 60% | structures correspond; a converter is a rename table |
| `MODERATE` | 15-60% | shapes largely align, module layout differs |
| `NOT-WORTH-IT` | < 15% | different graphs; a "converter" would be a reimplementation |
| `BLOCKED-BY-IMPORT` | reimpl would not construct | measurement impossible, cause recorded |
| `NO-CHECKPOINT` | no released weights exist | nothing to convert |

`pct_official_shape_recoverable_upper_bound` additionally counts official
tensors for which *some* unclaimed reimpl tensor has an identical shape. It
ignores whether such a pairing would be semantically correct, so it is a
strict **upper bound** on what any rename-only converter could ever recover.

