"""Fusion blocks: CIT, LC (attention + CSSM + gating), teacher, PAC, final."""

from .cit import CITModule, CITOutput
from .cssm import CSSM
from .lc import AttentionFusion, GatingUnit, LCModule
from .teacher import TeacherBranch
from .pac import BoxPositionalEmbedding, PACModule
from .adaptive import AdaptiveFusion

__all__ = ["CITModule", "CITOutput", "CSSM", "LCModule", "AttentionFusion",
           "GatingUnit", "TeacherBranch", "PACModule",
           "BoxPositionalEmbedding", "AdaptiveFusion"]
