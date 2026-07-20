"""CIT: mask exclusivity, sparsity, strategies, comm accounting."""

import torch

from cpbench.comms.channel import MessageChannel
from corabench.fusion.cit import CITModule


def _inputs(n_collab, c=8, h=6, w=6):
    ego_feat = torch.rand(c, h, w)
    ego_conf = torch.randn(1, h, w)
    feats = [torch.rand(c, h, w) for _ in range(n_collab)]
    confs = [torch.randn(1, h, w) for _ in range(n_collab)]
    return ego_feat, ego_conf, feats, confs


def test_winner_take_all_masks_exclusive():
    cit = CITModule("winner_take_all", request_threshold=0.0)
    ego_feat, ego_conf, feats, confs = _inputs(3)
    out = cit(ego_feat, ego_conf, feats, confs)
    total = torch.stack(out.masks).sum(dim=0)
    assert (total <= 1.0 + 1e-6).all()          # exclusive requests
    # f_coll equals the winning provider's feature at requested cells
    expected = sum(f * q for f, q in zip(feats, out.masks))
    assert torch.allclose(out.f_coll, expected)


def test_request_threshold_sparsifies():
    dense = CITModule("winner_take_all", request_threshold=0.0)
    sparse = CITModule("winner_take_all", request_threshold=0.4)
    args = _inputs(2)
    n_dense = sum(m.sum() for m in dense(*args).masks)
    n_sparse = sum(m.sum() for m in sparse(*args).masks)
    assert n_sparse < n_dense


def test_no_collaborators_degrades_gracefully():
    cit = CITModule()
    ego_feat, ego_conf, _, _ = _inputs(0)
    out = cit(ego_feat, ego_conf, [], [])
    assert out.f_coll.abs().sum() == 0 and (out.winner == -1).all()


def test_topk_and_maxout_shapes():
    for strategy in ("topk", "maxout"):
        cit = CITModule(strategy, topk=2)
        ego_feat, ego_conf, feats, confs = _inputs(3)
        out = cit(ego_feat, ego_conf, feats, confs)
        assert out.f_coll.shape == ego_feat.shape
        assert out.s_coll.shape == (1, *ego_feat.shape[1:])


def test_comm_volume_counted_and_sparse_cheaper():
    ego_feat, ego_conf, feats, confs = _inputs(2)
    ch_cit, ch_max = MessageChannel(), MessageChannel()
    CITModule("winner_take_all")(ego_feat, ego_conf, feats, confs,
                                 channel=ch_cit)
    CITModule("maxout")(ego_feat, ego_conf, feats, confs, channel=ch_max)
    assert ch_cit.log.total_bytes > 0
    feature_cit = ch_cit.log.bytes_by_location["channel/feature_msg"]
    feature_max = ch_max.log.bytes_by_location["channel/feature_msg"]
    assert feature_cit < feature_max            # on-demand < full broadcast
