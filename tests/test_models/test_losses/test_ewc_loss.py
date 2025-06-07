import torch
import torch.nn as nn

from mmseg.models.losses import EWCLoss


def test_ewc_loss():
    model = nn.Linear(3, 2)
    loss_fn = EWCLoss(ewc_lambda=0.5)
    # Fake fisher and previous params
    loss_fn.prev_params = {n: p.clone() for n, p in model.named_parameters()}
    loss_fn.fisher = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    loss = loss_fn(model)

    expected = 0
    for n, p in model.named_parameters():
        diff = p - loss_fn.prev_params[n]
        expected += (diff ** 2).sum()
    expected = expected * 0.5 * 0.5
    assert torch.allclose(loss, expected)