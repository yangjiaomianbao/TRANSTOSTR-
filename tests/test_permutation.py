import torch

from src.model.attention_probe import AttentionProbe


def test_permutation_equivariance(config):
    model = AttentionProbe(config).eval()
    x = torch.randn(2,30,96,2)
    perm = torch.randperm(30)
    with torch.no_grad():
        original, reordered = model(x), model(x[:,perm])
    for key in ("forecast", "node_tokens", "context_tokens"):
        torch.testing.assert_close(reordered[key],original[key][:,perm],atol=3e-6,rtol=3e-5)
    for key in ("attention_logits", "attention_weights"):
        torch.testing.assert_close(reordered[key],original[key][...,perm,:][...,:,perm],atol=3e-6,rtol=3e-5)
