"""
The two CoBEVT orchestrators and their non-attention parts.

Every operation is its own nn.Module and every forward takes
`taps: Optional[TapProtocol] = None`. No forward composes two operations in a
single expression -- see docs/cobevt_design.md section 5.4.

Contents
--------
cobevt_lidar   CoBEVTLidar: cpbench PointPillars + FuseBEVT + DetectionHead

sinbevt          SinBEVT: camera-to-BEV lifting, plus BEVSelfAttention
backbone         ResnetEncoder: multi-scale image features
decoder          NaiveDecoder: 32x32 BEV -> 256x256 segmentation grid
heads            BevSegHead: per-class segmentation logits
cobevt_camera    CoBEVTCamera: the paper's headline architecture
partition_hints  solve the query/key window constraint for the user
"""

from .backbone import ResnetEncoder
from .cobevt_camera import CoBEVTCamera
from .cobevt_lidar import CoBEVTLidar
from .decoder import NaiveDecoder
from .heads import BevSegHead
from .partition_hints import suggest_feature_window
from .sinbevt import BEVSelfAttention, SinBEVT

__all__ = ["CoBEVTCamera", "CoBEVTLidar", "SinBEVT", "BEVSelfAttention",
           "ResnetEncoder", "NaiveDecoder", "BevSegHead",
           "suggest_feature_window"]
