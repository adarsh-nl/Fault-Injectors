"""
info_quality
------------
Information-quality measurement for multi-platform, multi-modal fusion.

Information quality is operationalised as the mutual information I(Z; Y) between
a learned representation Z and a task-relevant target Y. Under fault injection,
tracking I(Z; Y) across severity levels measures how much task information a
representation retains as inputs degrade, and comparing the fused representation
against the single modalities measures whether fusion is synergistic.

The package separates the one model-specific step from the rest:

    feature_extraction : collect (Z, Y) arrays from ANY PyTorch model via
                         configurable forward hooks (the only model-aware part).
    preprocessing      : standardise / PCA / seed, applied identically to every
                         condition being compared.
    estimators         : InfoNCE and SMILE lower bounds on I(Z; Y). Pure
                         (N, d) array in, MI in nats out. Fully agnostic.
    reporting          : comparison tables and plots for arbitrary sets of
                         representations and estimators.
    run_mi             : CLI to estimate MI from a saved features file.

Everything except feature_extraction is independent of architecture and dataset.
"""

from .preprocessing import set_seed, standardise, pca_reduce, prepare
from .estimators import (
    MIResult,
    InfoNCEEstimator,
    SMILEEstimator,
    delta_information,
    correlated_gaussians,
)
from .feature_extraction import (
    FeatureCollector,
    FeatureSet,
    global_pool,
)
from .reporting import (
    format_table,
    fusion_summary,
    plot_comparison,
)

__all__ = [
    'set_seed', 'standardise', 'pca_reduce', 'prepare',
    'MIResult', 'InfoNCEEstimator', 'SMILEEstimator',
    'delta_information', 'correlated_gaussians',
    'FeatureCollector', 'FeatureSet', 'global_pool',
    'format_table', 'fusion_summary', 'plot_comparison',
]
