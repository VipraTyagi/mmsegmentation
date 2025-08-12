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
    def __init__(self,
                 ewc_lambda: float = 1_000.0,
                 fisher_path: str | None = None,
                 device: str = 'cuda'):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher: dict[str, torch.Tensor] = {}
        self.prev_params: dict[str, torch.Tensor] = {}
        
        
    def load_fisher(self, path: str,device: str = 'cuda') -> None:
        """Load previously computed Fisher information and parameters."""
        data = torch.load(path, map_location='cpu')
        
        
        
        
        # 1 ── ensure None‑checks and move to GPU/TPU if needed
        self.fisher = {k: v.to(device) for k, v in data.get('fisher', {}).items()}
        self.prev_params = {k: v.to(device) for k, v in data.get('params', {}).items()}
        print_log(f'Loaded Fisher information from {path}', logger='current')    
        
            # 2 ── shape / key check
        missing, mismatch = 0, 0
        for n, p in self.model.named_parameters():
            if n not in self.fisher:
                missing += 1
                continue
            if p.shape != self.fisher[n].shape:
                mismatch += 1
                raise RuntimeError(
                    f'Shape mismatch for {n}: model {tuple(p.shape)} vs '
                    f'fisher {tuple(self.fisher[n].shape)}')
        if missing:
            print_log(f'Fisher tensors loaded: {len(self.fisher)} '
                    f'( {missing} parameters not present in Fisher )',
                    logger='current')
        else:
            print_log(f'Loaded Fisher ({len(self.fisher)}) from {path}',
                    logger='current')

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
        if not self.fisher or not self.prev_params:
            print_log(
                'EWC loss skipped, fisher not initialized.', logger='current')
            return next(model.parameters()).new_tensor(0.)

        loss = next(model.parameters()).new_tensor(0.)
        for n, p in model.named_parameters():
            if n in self.fisher:
                diff = p - self.prev_params[n]
                loss = loss + (self.fisher[n] * diff.pow(2)).sum()

        loss = 0.5 * self.ewc_lambda * loss
        print_log(f'EWC loss value: {loss.item():.4f}', logger='current')
        return loss