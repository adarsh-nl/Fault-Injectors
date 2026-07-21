"""
cobevt_camera.py
----------------
The camera track: CoBEVT's headline architecture.

    images (B, L, M, H, W, 3)
      -> ResnetEncoder      -> three image feature scales
      -> SinBEVT            -> (N_total, 128, 32, 32)   per-agent BEV map
      -> NaiveCompressor    -> the bandwidth knob
      -> MessageChannel     -> byte accounting (no corruption)
      -> regroup            -> (B, L, 128, 32, 32) + agent mask
      -> SpatialTransform   -> ego-frame maps + validity mask
      -> FuseBEVT           -> (B, 128, 32, 32)
      -> NaiveDecoder       -> (B, 32, 256, 256)
      -> BevSegHead         -> (B, K, 256, 256)

Paper results this reproduces: 60.4 / 63.0 / 53.0 IoU for vehicle, drivable
area and lane, and the 44.3 IoU camera-dropout robustness figure that the
fault plane targets directly.

Two models, not one
-------------------
Assumption A8: the vehicle (``target="dynamic"``, 2 classes) and road/lane
(``target="static"``, 3 classes) results come from two separately trained
models merged at inference. This class builds one of them; the merge lives in
``evaluation/merge.py``. Building a single two-head model would be a
different training objective, and its numbers would not be comparable with
the paper's.

Where the faults land
---------------------
Every corruption reaches this model through the batch it is handed, never
through a hook inside it:

    camera dropout / occlusion / weather  ->  batch["images"]
    calibration error                     ->  batch["intrinsics"], ["extrinsics"]
    pose error                            ->  batch["T_agent_to_ego"] (the warp)
    agent drop                            ->  batch["record_len"] (the mask)
    latency                               ->  which frame each agent's data came from
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from cpbench.comms import MessageChannel
from cpbench.observation import TapProtocol, emit

from ..fusion.compression import NaiveCompressor
from ..fusion.fusebevt import FuseBEVT
from ..fusion.geometry import SpatialTransform, regroup
from .backbone import ResnetEncoder
from .decoder import NaiveDecoder
from .heads import BevSegHead
from .sinbevt import SinBEVT


class CoBEVTCamera(nn.Module):
    """CoBEVT's camera BEV semantic segmentation model.

    Purpose
        The paper's headline architecture, assembled from independently
        testable parts and instrumented at every intermediate tensor.

    Inputs
    ------
    target        ``"dynamic"`` or ``"static"`` (assumption A8)
    max_cav       fixed agent-axis extent (CoBEVT: 5)
    image_size    (H, W) of the input images (CoBEVT: 512, 512)
    bev_meters    metric extent of the BEV area (CoBEVT: 100 m)
    bev_size      BEV grid of SinBEVT's first block (CoBEVT: 128)
    dims, q_win_sizes, feat_win_sizes, heads, dim_head, middle,
    bev_embedding_flags   SinBEVT geometry, see that module
    backbone_arch, pretrained, id_pick   image backbone (assumption A2)
    compression   0 disables; 8/16/32/64 reproduce the paper's ablation
    fuse_*        FuseBEVT hyperparameters
    decoder_channels, upsample_mode      decoder (assumption A4)
    head_kernel_size                     assumption A3
    pool          assumption A11

    Outputs
    -------
    ``{"logits", "probs", "labels", "fused", "bev", "agent_mask"}``.
    ``bev`` is the per-agent transmitted map, returned so the compression
    ablation can be measured without re-running the encoder.

    Shapes
    ------
    batch["images"]         (N_total, M, H, W, 3)
    batch["intrinsics"]     (N_total, M, 3, 3)
    batch["extrinsics"]     (N_total, M, 4, 4)  camera -> that agent's ego frame
    batch["record_len"]     (B,) agents per sample, summing to N_total
    batch["T_agent_to_ego"] (B, max_cav, 4, 4)
    logits                  (B, K, bev_out * 2^n_decoder_stages, ...)

    Example
    -------
    >>> import torch
    >>> model = CoBEVTCamera(
    ...     target="dynamic", max_cav=2, image_size=(32, 32), bev_meters=40.0,
    ...     bev_size=16, dims=[16, 16], q_win_sizes=[8, 8],
    ...     feat_win_sizes=[2, 2], heads=[2, 2], dim_head=[8, 8],
    ...     middle=[1, 1], bev_embedding_flags=[True, False],
    ...     backbone_arch="resnet18", pretrained=False, id_pick=[1, 2],
    ...     fuse_window=4, fuse_dim_head=8, fuse_depth=1,
    ...     self_attn_dim_head=8, decoder_channels=[4, 8])
    >>> batch = {
    ...     "images": torch.rand(2, 4, 3, 32, 32),
    ...     "intrinsics": torch.eye(3).expand(2, 4, 3, 3).contiguous(),
    ...     "extrinsics": torch.eye(4).expand(2, 4, 4, 4).contiguous(),
    ...     "record_len": [2],
    ...     "T_agent_to_ego": torch.eye(4).expand(1, 2, 4, 4).contiguous(),
    ... }
    >>> out = model(batch)
    >>> out["logits"].shape, out["bev"].shape
    (torch.Size([1, 2, 32, 32]), torch.Size([2, 16, 8, 8]))
    """

    def __init__(self, target: str = "dynamic", max_cav: int = 5,
                 image_size: Tuple[int, int] = (512, 512),
                 bev_meters: float = 100.0, bev_size: int = 128,
                 dims: Sequence[int] = (128, 128, 128),
                 q_win_sizes: Sequence = (16, 16, 32),
                 feat_win_sizes: Sequence = (8, 8, 16),
                 heads: Sequence[int] = (4, 4, 4),
                 dim_head: Sequence[int] = (32, 32, 32),
                 middle: Sequence[int] = (2, 2, 2),
                 bev_embedding_flags: Sequence[bool] = (True, False, False),
                 backbone_arch: str = "resnet34", pretrained: bool = True,
                 id_pick: Sequence[int] = (1, 2, 3),
                 self_attn_dim_head: int = 32, dropout: float = 0.0,
                 camera_reduce: str = "mean", no_image_features: bool = False,
                 compression: int = 0,
                 fuse_depth: int = 3, fuse_window: int = 8,
                 fuse_dim_head: int = 32, fuse_mlp_dim: Optional[int] = None,
                 fuse_dropout: float = 0.0, use_local: bool = True,
                 use_global: bool = True, pool: str = "mean",
                 decoder_channels: Sequence[int] = (32, 64, 128),
                 upsample_mode: str = "nearest",
                 head_kernel_size: int = 3,
                 num_classes: Optional[int] = None) -> None:
        super().__init__()
        self.max_cav = int(max_cav)
        self.target = target
        self.bev_meters = float(bev_meters)

        self.backbone = ResnetEncoder(backbone_arch, pretrained, id_pick)
        if len(self.backbone.out_channels) != len(dims):
            raise ValueError(
                f"the backbone returns {len(self.backbone.out_channels)} "
                f"feature scales (id_pick={list(id_pick)}) but SinBEVT is "
                f"configured for {len(dims)} blocks; they must match")

        self.sinbevt = SinBEVT(
            dims=dims, feat_channels=self.backbone.out_channels,
            bev_size=bev_size, bev_meters=bev_meters, image_size=image_size,
            q_win_sizes=q_win_sizes, feat_win_sizes=feat_win_sizes,
            heads=heads, dim_head=dim_head, middle=middle,
            bev_embedding_flags=bev_embedding_flags,
            self_attn_dim_head=self_attn_dim_head, dropout=dropout,
            camera_reduce=camera_reduce, no_image_features=no_image_features)

        self._validate_window_plan(image_size, q_win_sizes, feat_win_sizes)

        bev_dim = self.sinbevt.out_dim
        bev_out = self.sinbevt.out_size
        if bev_out % fuse_window:
            raise ValueError(
                f"SinBEVT emits a {bev_out}x{bev_out} BEV map which does not "
                f"divide by the FuseBEVT window {fuse_window}; adjust "
                "bev_size, the number of SinBEVT blocks, or fuse_window")

        self.compressor = NaiveCompressor(bev_dim, compression)
        # The BEV grid is agent-centred, so the warp's origin is -extent/2 and
        # its stride is the metric size of one cell.
        stride = bev_meters / bev_out
        self.sttf = SpatialTransform(x_min=-bev_meters / 2.0,
                                     y_min=-bev_meters / 2.0,
                                     stride_x=stride, stride_y=stride)
        self.fuse = FuseBEVT(
            dim=bev_dim, mlp_dim=int(fuse_mlp_dim or 2 * bev_dim),
            agent_size=self.max_cav, window_size=fuse_window,
            dim_head=fuse_dim_head, dropout=fuse_dropout, depth=fuse_depth,
            use_local=use_local, use_global=use_global, pool=pool)
        self.decoder = NaiveDecoder(bev_dim, decoder_channels, upsample_mode)
        self.head = BevSegHead(target, decoder_channels[0], num_classes,
                               head_kernel_size)

    # -- eager validation ---------------------------------------------------

    @torch.no_grad()
    def _validate_window_plan(self, image_size: Tuple[int, int],
                              q_win_sizes: Sequence,
                              feat_win_sizes: Sequence) -> None:
        """Check the query/key window counts agree, before any data arrives.

        The constraint couples four independently configured numbers -- BEV
        size, query window, image resolution and key window -- so it is
        broken by changing any one of them alone. Left to runtime it surfaces
        from inside the cross-attention with the right complaint but at the
        wrong time: after the dataset is loaded and, on a cluster, after the
        job has been queued and started.

        The backbone's feature sizes are not knowable analytically for an
        arbitrary architecture, so they are probed with one dummy forward --
        the same trick the reference implementation uses to fill its
        ``output_shapes``. It costs one small forward pass at construction.
        """
        from .partition_hints import suggest_feature_window

        height, width = image_size
        probe = torch.zeros(1, 1, 3, int(height), int(width))
        was_training = self.training
        self.backbone.eval()
        feature_hw = [(f.shape[-2], f.shape[-1])
                      for f in self.backbone(probe)]
        self.backbone.train(was_training)

        problems = []
        for i, (bev, (feat_h, feat_w)) in enumerate(
                zip(self.sinbevt.bev_sizes, feature_hw)):
            q_win = q_win_sizes[i]
            q_win = (q_win, q_win) if isinstance(q_win, int) else tuple(q_win)
            k_win = feat_win_sizes[i]
            k_win = (k_win, k_win) if isinstance(k_win, int) else tuple(k_win)
            if bev % q_win[0] or bev % q_win[1]:
                problems.append(
                    f"  block {i}: BEV {bev}x{bev} does not divide by "
                    f"q_win_size {q_win}")
                continue
            if feat_h % k_win[0] or feat_w % k_win[1]:
                problems.append(
                    f"  block {i}: image features {feat_h}x{feat_w} do not "
                    f"divide by feat_win_size {k_win}")
                continue
            q_windows = (bev // q_win[0]) * (bev // q_win[1])
            k_windows = (feat_h // k_win[0]) * (feat_w // k_win[1])
            if q_windows != k_windows:
                suggestion = suggest_feature_window(bev, q_win, feat_h, feat_w)
                problems.append(
                    f"  block {i}: BEV {bev}x{bev} with q_win_size {q_win} "
                    f"gives {q_windows} query windows, but image features "
                    f"{feat_h}x{feat_w} with feat_win_size {k_win} give "
                    f"{k_windows}. Try feat_win_size {suggestion}."
                    if suggestion else
                    f"  block {i}: {q_windows} query windows vs {k_windows} "
                    f"key windows, and no feat_win_size divides "
                    f"{feat_h}x{feat_w} into {q_windows} windows -- change "
                    "q_win_size, bev_size or the image resolution instead.")
        if problems:
            raise ValueError(
                "SinBEVT cross-attention pairs query window i with key window "
                "i, so their counts must match at every block:\n"
                + "\n".join(problems))

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None,
                channel: Optional[MessageChannel] = None
                ) -> Dict[str, torch.Tensor]:
        record_len = [int(n) for n in batch["record_len"]]
        images = batch["images"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]

        emit(taps, images, module="CoBEVTCamera", location="input/images")
        emit(taps, intrinsics, module="CoBEVTCamera",
             location="input/intrinsics")
        emit(taps, extrinsics, module="CoBEVTCamera",
             location="input/extrinsics")

        features = self.backbone(images, taps=taps)
        bev = self.sinbevt(features, intrinsics, extrinsics, taps=taps)
        bev = self.compressor(bev, taps=taps)
        bev = self._account(bev, record_len, channel, taps)

        padded, agent_mask = regroup(bev, record_len, self.max_cav, taps=taps)
        emit(taps, agent_mask, module="CoBEVTCamera",
             location="input/agent_mask")

        transforms = batch["T_agent_to_ego"].to(padded.dtype)
        emit(taps, transforms, module="CoBEVTCamera", location="input/poses")
        warped, valid = self.sttf(padded, transforms, taps=taps)

        fuse_mask = agent_mask[:, :, None, None] & valid
        fused = self.fuse(warped, mask=fuse_mask, taps=taps)

        decoded = self.decoder(fused, taps=taps)
        out = self.head(decoded, taps=taps)
        out["fused"] = fused
        out["bev"] = bev
        out["agent_mask"] = fuse_mask
        return out

    def _account(self, bev: torch.Tensor, record_len: Sequence[int],
                 channel: Optional[MessageChannel],
                 taps: Optional[TapProtocol]) -> torch.Tensor:
        """Log transmitted bytes per collaborator. Never modifies the tensor.

        The ego does not transmit to itself, so it is excluded -- counting it
        would overstate the communication volume by one agent per scene and
        make the compression ablation's KB figures wrong.
        """
        if channel is None:
            emit(taps, bev, module="MessageChannel", location="comm/sent")
            return bev
        offset = 0
        for sample, count in enumerate(record_len):
            for local_index in range(1, count):          # skip ego at 0
                index = offset + local_index
                channel.send(bev[index], sender=f"s{sample}a{local_index}",
                             receiver=f"s{sample}ego", location="comm/sent")
            offset += count
        return bev

    def extra_repr(self) -> str:
        return (f"target={self.target}, max_cav={self.max_cav}, "
                f"bev={self.sinbevt.out_dim}x{self.sinbevt.out_size}"
                f"x{self.sinbevt.out_size}")
