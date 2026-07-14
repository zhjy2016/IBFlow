import torch
import torch.nn as nn
from functools import partial
from dataclasses import dataclass
from mmcv.cnn import constant_init, xavier_init
from diffusers.utils import BaseOutput


@dataclass
class IBFlowModelOutput(BaseOutput):
    """
    The output of IBFlow models.

    Args:
        means (`torch.Tensor` of shape `(batch_size, num_gaussians, num_channels, height, width)` or
        `(batch_size, num_gaussians, num_channels, frame, height, width)`):
            Gaussian mixture means.
        logweights (`torch.Tensor` of shape `(batch_size, num_gaussians, 1, height, width)` or
        `(batch_size, num_gaussians, 1, frame, height, width)`):
            Gaussian mixture log-weights (logits).
        """

    means: torch.Tensor
    logweights: torch.Tensor
    loggammas: torch.Tensor

