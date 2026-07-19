"""
pac.py
------
Pose-Aware Correction (PAC) module -- paper Fig. 4, Eqs. 12-16.

The object-level branch: collaborators transmit their LOCAL detection maps
O_j = (C_j, R_j); PAC rectifies the spatial misalignment that upstream pose
error imprinted on them, using two complementary mechanisms:

    semantic   descriptors from positional embeddings of decoded box
               parameters -> cross-agent attention map A_j -> relevance
               scoring (Eqs. 12-13)
    geometric  dense offset field Delta-p_j -> deformable convolution
               resampling (Eqs. 14-16)

and fuses the two corrected representations (assumption A3: 1x1 conv over
the channel concat). Multiple collaborators are merged cell-wise by best
corrected confidence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torchvision.ops import deform_conv2d

from ..observation.taps import TapProtocol, emit


class BoxPositionalEmbedding(nn.Module):
    """Sinusoidal embedding of per-cell decoded box parameters (Eq. 12, A7).

    Purpose  turn each cell's detection hypothesis
             (x, y, z, l, h, w, alpha, delta) into a smooth descriptor whose
             similarity reflects geometric + confidence closeness.
    Inputs   params (B, 8, H, W) -- decoded box parameters + confidence.
    Output   (B, out_dim, H, W) descriptor.

    Each of the 8 parameters gets `freqs` sine/cosine pairs (dim 16*freqs),
    projected to `out_dim` by a 1x1 conv.
    """

    def __init__(self, out_dim: int = 64, freqs: int = 4) -> None:
        super().__init__()
        self.freqs = freqs
        self.register_buffer(
            "bands", 2.0 ** torch.arange(freqs).float() * math.pi)
        self.proj = nn.Conv2d(8 * 2 * freqs, out_dim, 1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        b, p, h, w = params.shape
        x = params.unsqueeze(2) * self.bands.view(1, 1, -1, 1, 1)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=2)   # (B, 8, 2F, H, W)
        return self.proj(enc.reshape(b, p * 2 * self.freqs, h, w))


class PACModule(nn.Module):
    """Object-level correction of collaborator detection maps.

    Inputs (one ego frame at a time; conv batch dim = collaborators)
    ------
    ego_maps     (cls_i, reg_i): (Ncls_ch, H, W), (Nreg_ch, H, W) -- the
                 ego's LOCAL head outputs (Ncls_ch = A*num_classes,
                 Nreg_ch = A*7).
    collab_maps  list of (cls_j, reg_j) with identical shapes.
    anchors      (H, W, A, 7) buffer for per-cell box decoding.

    Output
    ------
    (cls_pac, reg_pac): corrected + merged collaborator maps, same shapes as
    the inputs; zeros when there are no collaborators.

    Example
    -------
    >>> pac = PACModule(ncls_ch=2, nreg_ch=14, anchors=anchors)  # doctest: +SKIP
    >>> cls_pac, reg_pac = pac((ci, ri), [(cj, rj)])             # doctest: +SKIP
    """

    def __init__(self, ncls_ch: int, nreg_ch: int, anchors: torch.Tensor,
                 pe_dim: int = 64, kernel_size: int = 3,
                 select_hidden: int = 32) -> None:
        super().__init__()
        self.ncls_ch, self.nreg_ch = ncls_ch, nreg_ch
        self.k = kernel_size
        self.num_anchors = anchors.shape[2]
        self.register_buffer("anchors", anchors.float(), persistent=False)

        io = ncls_ch + nreg_ch
        self.select = nn.Sequential(                       # high-conf selection
            nn.Conv2d(io, select_hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(select_hidden, 1, 1), nn.Sigmoid())
        self.pe = BoxPositionalEmbedding(pe_dim)
        self.f_attn = nn.Sequential(                       # Eq. 12
            nn.Conv2d(2 * pe_dim, pe_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pe_dim, 1, 1))
        self.f_offset = nn.Sequential(                     # Eq. 14
            nn.Conv2d(2 * io, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1))
        self.deform_weight_cls = nn.Parameter(
            torch.randn(ncls_ch, ncls_ch, kernel_size, kernel_size) * 0.01)
        self.deform_weight_reg = nn.Parameter(
            torch.randn(nreg_ch, nreg_ch, kernel_size, kernel_size) * 0.01)
        with torch.no_grad():                              # init near identity
            eye = torch.eye(ncls_ch)
            self.deform_weight_cls[:, :, kernel_size // 2, kernel_size // 2] = eye
            eye = torch.eye(nreg_ch)
            self.deform_weight_reg[:, :, kernel_size // 2, kernel_size // 2] = eye
        self.fuse_cls = nn.Conv2d(2 * ncls_ch, ncls_ch, 1)   # A3
        self.fuse_reg = nn.Conv2d(2 * nreg_ch, nreg_ch, 1)

    # -- helpers ------------------------------------------------------------

    def _decode_params(self, cls_map: torch.Tensor,
                       reg_map: torch.Tensor) -> torch.Tensor:
        """Per-cell (x, y, z, l, h, w, alpha, delta) of the best anchor.

        cls_map (B, A*ncls, H, W), reg_map (B, A*7, H, W) -> (B, 8, H, W).
        Inverse of the TargetAssigner encoding, evaluated at the
        highest-confidence anchor of each cell.
        """
        b, _, h, w = cls_map.shape
        a = self.num_anchors
        scores = torch.sigmoid(cls_map).reshape(b, a, -1, h, w).amax(dim=2)
        best = scores.argmax(dim=1)                              # (B, H, W)
        conf = scores.amax(dim=1)
        reg = reg_map.reshape(b, a, 7, h, w)
        reg = reg.gather(1, best[:, None, None].expand(-1, 1, 7, -1, -1))[:, 0]
        anch = self.anchors.unsqueeze(0).expand(b, -1, -1, -1, -1)  # (B,H,W,A,7)
        anch = anch.gather(3, best[..., None, None].expand(b, h, w, 1, 7))
        anch = anch.squeeze(3).permute(0, 3, 1, 2)               # (B, 7, H, W)
        d = (anch[:, 3] ** 2 + anch[:, 4] ** 2).sqrt()
        x = reg[:, 0] * d + anch[:, 0]
        y = reg[:, 1] * d + anch[:, 1]
        z = reg[:, 2] * anch[:, 5] + anch[:, 2]
        lwh = torch.exp(reg[:, 3:6].clamp(-5, 5)) * anch[:, 3:6]
        alpha = anch[:, 6] + torch.asin(reg[:, 6].clamp(-1, 1))
        return torch.stack([x, y, z, lwh[:, 0], lwh[:, 2], lwh[:, 1],
                            alpha, conf], dim=1)                 # (B, 8, H, W)

    # -- forward ------------------------------------------------------------

    def forward(self, ego_maps: Tuple[torch.Tensor, torch.Tensor],
                collab_maps: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                taps: Optional[TapProtocol] = None,
                frame: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_i, reg_i = ego_maps
        if len(collab_maps) == 0:
            return (torch.zeros_like(cls_i), torch.zeros_like(reg_i))

        cls_j = torch.stack([m[0] for m in collab_maps])     # (N, Ccls, H, W)
        reg_j = torch.stack([m[1] for m in collab_maps])     # (N, Creg, H, W)
        n = cls_j.shape[0]
        ego_cls = cls_i.unsqueeze(0).expand(n, -1, -1, -1)
        ego_reg = reg_i.unsqueeze(0).expand(n, -1, -1, -1)

        # high-confidence selection
        sel = self.select(torch.cat([cls_j, reg_j], dim=1))  # (N, 1, H, W)
        cls_s, reg_s = cls_j * sel, reg_j * sel
        emit(taps, torch.cat([cls_s, reg_s], dim=1), module="PACModule",
             location="pac/selected_collab", frame=frame)

        # semantic branch: PE descriptors -> attention map -> scoring
        pe_i = self.pe(self._decode_params(ego_cls, ego_reg))
        pe_j = self.pe(self._decode_params(cls_s, reg_s))
        emit(taps, pe_i, module="PACModule", location="pac/pe_ego", frame=frame)
        emit(taps, pe_j, module="PACModule", location="pac/pe_collab",
             frame=frame)
        attn = torch.sigmoid(self.f_attn(torch.cat([pe_i, pe_j], dim=1)))
        emit(taps, attn, module="PACModule", location="pac/attention_map",
             frame=frame)
        cls_p = cls_s * attn                                 # Eq. 13
        reg_p = reg_s * attn
        emit(taps, cls_p, module="PACModule", location="pac/scored_cls",
             frame=frame)
        emit(taps, reg_p, module="PACModule", location="pac/scored_reg",
             frame=frame)

        # geometric branch: offset field -> deformable resampling
        offset2 = self.f_offset(
            torch.cat([ego_cls, ego_reg, cls_s, reg_s], dim=1))  # (N, 2, H, W)
        emit(taps, offset2, module="PACModule", location="pac/offset_field",
             frame=frame)
        offsets = offset2.repeat(1, self.k * self.k, 1, 1)   # same shift per tap
        pad = self.k // 2
        cls_pp = deform_conv2d(cls_s, offsets, self.deform_weight_cls,
                               padding=pad)                  # Eq. 15
        reg_pp = deform_conv2d(reg_s, offsets, self.deform_weight_reg,
                               padding=pad)                  # Eq. 16
        emit(taps, cls_pp, module="PACModule", location="pac/corrected_cls",
             frame=frame)
        emit(taps, reg_pp, module="PACModule", location="pac/corrected_reg",
             frame=frame)

        # fuse semantic + geometric corrections (A3)
        cls_out = self.fuse_cls(torch.cat([cls_p, cls_pp], dim=1))
        reg_out = self.fuse_reg(torch.cat([reg_p, reg_pp], dim=1))

        # merge collaborators: per-cell winner by corrected confidence
        conf = torch.sigmoid(cls_out).amax(dim=1)            # (N, H, W)
        winner = conf.argmax(dim=0)                          # (H, W)
        h, w = winner.shape
        idx_c = winner.view(1, 1, h, w).expand(1, cls_out.shape[1], h, w)
        idx_r = winner.view(1, 1, h, w).expand(1, reg_out.shape[1], h, w)
        cls_final = cls_out.gather(0, idx_c)[0]
        reg_final = reg_out.gather(0, idx_r)[0]
        emit(taps, cls_final, module="PACModule", location="pac/output_cls",
             frame=frame)
        emit(taps, reg_final, module="PACModule", location="pac/output_reg",
             frame=frame)
        return cls_final, reg_final
