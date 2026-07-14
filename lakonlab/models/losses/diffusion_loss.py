# Copyright (c) 2025 Hansheng Chen

from functools import partial

from mmgen.models import MODULES
from mmgen.models.losses.ddpm_loss import DDPMLoss, mse_loss


@MODULES.register_module()
class DiffusionMSELoss(DDPMLoss):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 sampler=None,
                 weight=None,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_mse'):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss
