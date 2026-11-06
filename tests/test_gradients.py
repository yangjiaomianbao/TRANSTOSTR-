import torch
import torch.nn.functional as F

from src.model.attention_probe import AttentionProbe


def test_all_paths_have_gradients_and_optimizer_updates(config):
    model = AttentionProbe(config)
    phase = torch.randn(2,30,128)
    x = torch.stack((phase.sin(),phase.cos()),dim=-1)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    result = model(x[:,:,:96])
    result["attention_logits"].retain_grad()
    # Saved logits are differentiable even though concatenation is an export view.
    assert result["attention_logits"].requires_grad
    loss = F.mse_loss(result["forecast"],x[:,:,96:])
    loss.backward()
    groups = [model.tcn, model.forecast_head]
    for layer in model.layers:
        groups.extend([layer.q,layer.k,layer.v,layer.ffn])
    for group in groups:
        grads = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads)>0
    old = model.layers[0].q.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old,model.layers[0].q.weight)
