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

    def __init__(self, ewc_lambda: float = 1.0, dataloader=None) -> None:
        self.ewc = EWCLoss(ewc_lambda=ewc_lambda)
        self.dataloader = dataloader

    def before_train(self, runner) -> None:
        """Estimate Fisher information before training if dataloader given."""
        if self.dataloader is not None:
            self.ewc.update_fisher(runner.model, self.dataloader)
            print_log('Fisher information estimated.', logger='current')

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None) -> None:
        """Add the EWC penalty to the loss of each iteration."""
        penalty = self.ewc(runner.model)
        if outputs is None:
            outputs = runner.outputs
        if isinstance(outputs, dict) and 'loss' in outputs:
            outputs['loss'] = outputs['loss'] + penalty
            print_log(
                f'EWC penalty: {penalty.item():.4f}, total loss: '
                f'{outputs["loss"].item():.4f}',
                logger='current')