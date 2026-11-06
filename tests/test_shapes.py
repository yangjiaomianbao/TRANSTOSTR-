import pytest
import torch

from src.model.attention_probe import AttentionProbe


@pytest.mark.parametrize("batch", [1, 3])
def test_shapes(config, batch):
    model = AttentionProbe(config).eval()
    x = torch.randn(batch, 30, 96, 2)
    with torch.no_grad():
        h = model.tcn(x)
        result = model(x)
    assert h.shape == (batch, 30, 96, 64)
    assert model.tcn.receptive_field == 31
    assert result["node_tokens"].shape == (batch, 30, 64)
    assert result["context_tokens"].shape == (batch, 30, 64)
    assert result["forecast"].shape == (batch, 30, 32, 2)
    torch.testing.assert_close(result["forecast"].square().sum(-1), torch.ones(batch,30,32))
    for key in ("attention_logits", "attention_weights"):
        assert result[key].shape == (batch, 2, 4, 30, 30)
        assert torch.isfinite(result[key]).all()
    a = result["attention_weights"]
    torch.testing.assert_close(a.sum(-1), torch.ones(batch,2,4,30))
    assert torch.count_nonzero(a.diagonal(dim1=-2, dim2=-1)) == 0


def test_node_only_and_cross_node_dependence(config):
    x = torch.randn(1, 30, 96, 2)
    changed = x.clone(); changed[:,1:] += 3
    node = AttentionProbe(config, node_only=True).eval()
    full = AttentionProbe(config).eval()
    with torch.no_grad():
        a, b = node(x), node(changed)
        fa, fb = full(x), full(changed)
    torch.testing.assert_close(a["forecast"][:,0], b["forecast"][:,0])
    assert a["attention_logits"] is None
    assert not torch.allclose(fa["forecast"][:,0], fb["forecast"][:,0])
    ratio = sum(p.numel() for p in node.parameters()) / sum(p.numel() for p in full.parameters())
    assert .9 < ratio < 1.1
