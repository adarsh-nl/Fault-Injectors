"""Named tap locations for the measurement plane (cpbench.observation).

The heads already emit head/cls_logits, head/reg_map, head/cls_sigmoid per
branch; these names document the package-specific seams a robustness
question inspects.
"""
LOCATIONS = (
    "encoder/bev_features",
    "cit/relevance", "cit/masks", "cit/collab_feature",
    "lc/z_fused", "lc/ssm_out", "lc/output",
    "pac/gate_log", "pac/offsets", "pac/output_cls",
    "fusion/uncertainty_lc", "fusion/uncertainty_pac",
    "head/cls_logits", "head/reg_map",
)
