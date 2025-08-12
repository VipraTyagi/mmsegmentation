import logging
from mmengine.hooks import Hook
from mmseg.models.losses import EWCLoss
from mmseg.registry import HOOKS
from mmengine.logging import print_log

@HOOKS.register_module()
class EWCHook(Hook):
    """Hook to apply Elastic Weight Consolidation during training.

    Args:
        ewc_lambda (float): Strength of the EWC penalty. Defaults to 1.0.
        dataloader (DataLoader, optional): Data loader providing samples from
            the previous task used to compute the Fisher information. If
            provided, the Fisher matrix will be estimated at the beginning of
            training.
    """

    def __init__(self, ewc_lambda: float = 1.0, fisher_path: str | None = None,dataloader=None) -> None:
        self.ewc = EWCLoss(ewc_lambda=ewc_lambda, fisher_path=fisher_path)
        self.dataloader = dataloader
        self.fisher_path = fisher_path

    def before_train(self, runner) -> None:
        """Load or estimate Fisher information before training."""
        if self.fisher_path is not None:
            self.ewc.model = runner.model
            self.ewc.load_fisher(self.fisher_path, device='cuda')
            print_log(
                f'Fisher information loaded from {self.fisher_path}.',
                logger='current')
        elif self.dataloader is not None:
            self.ewc.update_fisher(runner.model, self.dataloader)
            print_log('Fisher information estimated.', logger='current')

        if not self.ewc.fisher:
            print_log(
                'No Fisher information found. EWC penalty will be zero.',
                logger='current',
                level=logging.WARNING)
        else:
            model_names = {
                n
                for n, p in runner.model.named_parameters()
                if p.requires_grad
            }
            fisher_names = set(self.ewc.prev_params.keys())
            if model_names != fisher_names:
                missing = sorted(model_names - fisher_names)
                extra = sorted(fisher_names - model_names)
                print_log(
                    'Mismatch between Fisher parameters and current '
                    f'model. missing: {missing} extra: {extra}',
                    logger='current',
                    level=logging.WARNING)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None) -> None:
        """Add the EWC penalty to the loss of each iteration."""      
        if outputs is None:
            outputs = runner.outputs
        penalty = self.ewc(runner.model)
        if isinstance(outputs, dict) and 'loss' in outputs:
            outputs['loss'] = outputs['loss'] + penalty
            print_log(
                f'EWC penalty: {penalty.item():.4f}, total loss: '
                f'{outputs["loss"].item():.4f}',
                logger='current')
        else:
        # Some custom heads return a tensor; handle that too
            outputs['loss'] = outputs.get('loss', 0.0) + penalty
            
            
        if runner.iter % 50 == 0:
            ce = None
            # MMSeg heads typically expose one of these keys
            for k in ('decode.loss_ce', 'loss_ce', 'aux.loss_ce'):
                if k in outputs:
                    ce = outputs[k]
                    break

            if ce is not None and ce.item() > 0:
                ratio = penalty.item() / (ce.item() + 1e-8)
                print_log(f'EWC/CE ratio: {ratio:.3f}', logger='current')
            else:
                # Debug helper: show what keys are available
                print_log(f'CE term not found in outputs. Keys: {list(outputs.keys())}',
                        logger='current')