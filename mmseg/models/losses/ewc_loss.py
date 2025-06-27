import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmseg.registry import MODELS

@MODELS.register_module()
class EWCLoss(nn.Module):
    """Elastic Weight Consolidation loss.

    This loss implements the regularization term described in
    "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al.
    It penalizes deviations from parameters important to previous tasks using
    their estimated Fisher information.

    Args:
        ewc_lambda (float): Scaling factor for the EWC penalty. Defaults to 1.0.
    """

    def __init__(self, ewc_lambda: float = 1.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher = None
        self.prev_params = None

    @torch.no_grad()
    def update_fisher(self, model: nn.Module, dataloader) -> None:
        """Estimate diagonal Fisher information for current ``model``.

        The model will be evaluated on ``dataloader`` and the squared gradients
        of the loss w.r.t. each parameter are accumulated as an importance
        estimate. After calling this method, the current parameters are stored as
        reference parameters for later regularization.
        """
        self.prev_params = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters() if p.requires_grad
        }

        was_training = model.training
        model.eval()
        for data in dataloader:
            outputs = model(**data)
            loss = outputs if torch.is_tensor(outputs) else outputs['loss']
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach()**2
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)
        print_log(
            'Fisher information updated with mean values: ' +
            ', '.join(
                f'{k}:{v.mean().item():.4f}' for k, v in self.fisher.items()),
            logger='current')
        model.train(was_training)

    def forward(self, model: nn.Module) -> torch.Tensor:
            """Calculate the EWC regularization term for ``model``."""
            if self.fisher is None or self.prev_params is None:
                print_log('EWC loss skipped, fisher not initialized.', logger='current')
                return next(model.parameters()).new_tensor(0.)
            loss = next(model.parameters()).new_tensor(0.)
            for n, p in model.named_parameters():
                if n in self.fisher:
                    diff = p - self.prev_params[n]
                    loss = loss + (self.fisher[n] * diff.pow(2)).sum()
            loss = 0.5 * self.ewc_lambda * loss
            print_log(f'EWC loss value: {loss.item():.4f}', logger='current')
            return loss