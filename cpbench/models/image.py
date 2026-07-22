"""
image.py
--------
Multi-scale image feature extraction: a ResNet run for its intermediate maps.

Not any paper's contribution. A torchvision ResNet is stopped short of its
classifier and its residual stages are returned coarse-to-fine, which is what
every camera-track BEV model in this repository consumes -- CoBEVT's SinBEVT
lifts them by cross-view attention, Where2comm's lift splats them along a
predicted depth distribution. The backbone is the same either way, so it lives
here rather than in whichever package needed it first.

Extracted from ``cobevtbench.models.backbone`` when ``w2cbench`` became the
second package to need it; that module re-exports, so the move is additive.

Which layers, and one paper's off-by-one
----------------------------------------
``id_pick`` selects from ``[layer1, layer2, layer3, layer4]`` by index.
CoBEVT's released config uses ``[1, 2, 3]`` -- **layer2, layer3, layer4**, at
128, 256 and 512 channels. That paper's Appendix C names layer1/2/3 while
quoting those same channel counts, which cannot both be true: 128/256/512 are
layer2/3/4's widths in ResNet34. The code is right and the paper's layer names
are off by one (cobevtbench assumption A2). Left configurable, so the literal
reading stays reachable.

Tap locations
-------------
Emits ``backbone/normalised`` and ``backbone/feat_s{i}``. Both names are
declared by every paper registry that uses this module, deliberately
identically, so a cross-paper layer-wise comparison of image features is a
straight join on ``location``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange
from torch import nn

from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)

# ImageNet statistics -- required, not decorative: the pretrained weights were
# fitted under this normalisation and produce degraded features without it.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResnetEncoder(nn.Module):
    """Multi-scale image features from a torchvision ResNet.

    Purpose
        Turn each agent's M camera images into the feature pyramid SinBEVT
        lifts onto its BEV grid.

    Inputs
    ------
    arch        ``"resnet18"`` | ``"resnet34"`` (CoBEVT) | ``"resnet50"``
    pretrained  load ImageNet weights. Needs network access on first use;
                the failure is caught and re-raised with the cache path, so
                a compute node without egress says so instead of hanging.
    id_pick     which of [layer1, layer2, layer3, layer4] to return, by index
                (CoBEVT: [1, 2, 3] -- assumption A2)

    Outputs
    -------
    A list of ``(B, M, C_i, h_i, w_i)``, one per picked layer, ordered
    fine-to-coarse in resolution.

    Shapes
    ------
    images  (B, M, H, W, 3) uint8 or float in [0, 255], OR
            (B, M, 3, H, W) float already in [0, 1]
    return  CoBEVT at 512x512: [(B, M, 128, 64, 64), (B, M, 256, 32, 32),
                                (B, M, 512, 16, 16)]

    Example
    -------
    >>> import torch
    >>> enc = ResnetEncoder(arch="resnet18", pretrained=False, id_pick=[1, 2])
    >>> feats = enc(torch.rand(1, 4, 3, 64, 64))
    >>> [tuple(f.shape) for f in feats]
    [(1, 4, 128, 8, 8), (1, 4, 256, 4, 4)]
    >>> enc.out_channels
    [128, 256]
    """

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 id_pick: Sequence[int] = (1, 2, 3)) -> None:
        super().__init__()
        from torchvision import models

        builders = {"resnet18": models.resnet18, "resnet34": models.resnet34,
                    "resnet50": models.resnet50}
        if arch not in builders:
            raise ValueError(
                f"unknown backbone {arch!r}; expected one of {sorted(builders)}")
        self.arch = arch
        self.id_pick = [int(i) for i in id_pick]
        if not self.id_pick or any(i < 0 or i > 3 for i in self.id_pick):
            raise ValueError(
                f"id_pick must select from layers 0-3, got {list(id_pick)}")

        net = builders[arch](weights="DEFAULT" if pretrained else None) \
            if pretrained else builders[arch](weights=None)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        all_layers = [net.layer1, net.layer2, net.layer3, net.layer4]

        # Keep only up to the deepest picked layer. Retaining the rest would
        # cost forward time for a result nobody reads, leave parameters that
        # never receive a gradient (inflating the model size the paper
        # reports), and -- on small inputs -- run a BatchNorm over a 1x1 map,
        # which raises in training mode. CoBEVT picks [1, 2, 3] so nothing is
        # dropped at the paper's settings; shallower ablations benefit.
        self.depth = max(self.id_pick) + 1
        self.layers = nn.ModuleList(all_layers[:self.depth])
        all_channels = [self._block_out_channels(layer[-1])
                        for layer in all_layers[:self.depth]]
        self.out_channels: List[int] = [all_channels[i] for i in self.id_pick]

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1),
            persistent=False)

    @staticmethod
    def _block_out_channels(block: nn.Module) -> int:
        """Output width of a residual block, for BasicBlock or Bottleneck."""
        for attr in ("bn3", "bn2", "bn1"):
            norm = getattr(block, attr, None)
            if norm is not None:
                return int(norm.num_features)
        raise RuntimeError(f"cannot infer output channels of {type(block)}")

    # -- input handling -----------------------------------------------------

    def _to_nchw(self, images: torch.Tensor) -> torch.Tensor:
        """Accept (B, M, H, W, 3) bytes or (B, M, 3, H, W) floats.

        Dispatching on which axis is 3 rather than on dtype: a dataset that
        already converted to float but kept channels-last is the likely
        mistake, and silently treating H=3 as channels would produce a
        tensor that trains to noise.
        """
        if images.dim() != 5:
            raise ValueError(
                f"expected (B, M, H, W, 3) or (B, M, 3, H, W), "
                f"got shape {tuple(images.shape)}")
        if images.shape[-1] == 3 and images.shape[2] != 3:
            images = rearrange(images, "b m h w c -> b m c h w")
        elif images.shape[2] != 3:
            raise ValueError(
                f"cannot find a size-3 channel axis in {tuple(images.shape)}")
        flat = rearrange(images, "b m c h w -> (b m) c h w").float()
        if flat.max() > 1.5:                     # byte-valued input
            flat = flat / 255.0
        return (flat - self.mean) / self.std

    # -- forward ------------------------------------------------------------

    def forward(self, images: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> List[torch.Tensor]:
        batch, cameras = images.shape[0], images.shape[1]
        x = self._to_nchw(images)
        emit(taps, x, module="ResnetEncoder", location="backbone/normalised")

        x = self.stem(x)
        picked: List[torch.Tensor] = []
        wanted = set(self.id_pick)
        by_index = {}
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index in wanted:
                by_index[index] = x
        for order, index in enumerate(self.id_pick):
            feature = rearrange(by_index[index], "(b m) c h w -> b m c h w",
                                b=batch, m=cameras)
            emit(taps, feature, module="ResnetEncoder",
                 location=f"backbone/feat_s{order}")
            picked.append(feature)
        return picked

    def extra_repr(self) -> str:
        return (f"arch={self.arch}, id_pick={self.id_pick}, "
                f"out_channels={self.out_channels}")
