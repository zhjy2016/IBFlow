# Copyright (c) 2025 Hansheng Chen

import math
import torch

from typing import Dict
from .base import BasePolicy


class IBFlowPolicy(BasePolicy):
    """IBFlow policy. The component count is inferred from the denoising output.

    Args:
        denoising_output (dict): The output of the denoising model, containing:
            means (torch.Tensor): The means of the Gaussian components. Shape (B, K, C, H, W) or (B, K, C, T, H, W).
            logstds (torch.Tensor): The log standard deviations of the Gaussian components. Shape (B, K, 1, 1, 1)
                or (B, K, 1, 1, 1, 1).
            logweights (torch.Tensor): The log weights of the Gaussian components. Shape (B, K, 1, H, W) or
                (B, K, 1, T, H, W).
        x_t_src (torch.Tensor): The initial noisy sample. Shape (B, C, H, W) or (B, C, T, H, W).
        sigma_t_src (torch.Tensor): The initial noise level. Shape (B,).
        checkpointing (bool): Whether to use gradient checkpointing to save memory. Defaults to True.
        eps (float): A small value to avoid numerical issues. Defaults to 1e-4.
    """

    def __init__(
            self,
            denoising_output: Dict[str, torch.Tensor],
            x_t_src: torch.Tensor,
            sigma_t_src: torch.Tensor,
            checkpointing: bool = True,
            eps: float = 1e-4):
        self.x_t_src = x_t_src
        self.ndim = x_t_src.dim()
        self.checkpointing = checkpointing
        self.eps = eps

        self.sigma_t_src = sigma_t_src.reshape(*sigma_t_src.size(), *((self.ndim - sigma_t_src.dim()) * [1]))
        self.denoising_output_x_0 = self._u_to_x_0(
            denoising_output, self.x_t_src, self.sigma_t_src)

    @staticmethod
    def _u_to_x_0(denoising_output, x_t, sigma_t):
        x_t = x_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)
        means_x_0 = x_t - sigma_t * denoising_output['means']
        return dict(
            means=means_x_0,
            logweights=denoising_output['logweights'],
            loggammas=denoising_output.get('loggammas', None),
            means_u=denoising_output['means'])

    def velocity(self, sigma_t_src, sigma_t):
        """Compute the flow velocity at (t).

        Args:
            sigma_t_src (torch.Tensor): Noise level at source time t_src.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        means = self.denoising_output_x_0['means_u']
        log_gammas = self.denoising_output_x_0['loggammas']
        logweights = self.denoising_output_x_0['logweights']
        weights = torch.softmax(logweights, dim=1)  
        dt_past = sigma_t_src - sigma_t
        
        dt_past = dt_past.unsqueeze(1)  
        decay_factor = torch.exp(log_gammas * dt_past)
        bs, k, c, h, w = decay_factor.shape
        decay_factor = torch.cat(
            [torch.ones((bs, 1, c, h, w), device=decay_factor.device),
             decay_factor],
            dim=1)
        v_at_a = (means * decay_factor * weights).sum(dim=1)
        return v_at_a

    def copy(self):
        new_policy = IBFlowPolicy.__new__(IBFlowPolicy)
        new_policy.x_t_src = self.x_t_src
        new_policy.ndim = self.ndim
        new_policy.checkpointing = self.checkpointing
        new_policy.eps = self.eps
        new_policy.sigma_t_src = self.sigma_t_src
        new_policy.denoising_output_x_0 = self.denoising_output_x_0.copy()
        return new_policy

    def detach_(self):
        self.denoising_output_x_0 = {k: v.detach() for k, v in self.denoising_output_x_0.items()}
        return self

    def detach(self):
        new_policy = self.copy()
        return new_policy.detach_()

    def dropout_(self, p):
        if p <= 0 or p >= 1:
            return self
        logweights = self.denoising_output_x_0['logweights']
        dropout_mask = torch.rand(
            (*logweights.shape[:2], *((self.ndim - 1) * [1])), device=logweights.device) < p
        is_all_dropout = dropout_mask.all(dim=1, keepdim=True)
        dropout_mask &= ~is_all_dropout
        self.denoising_output_x_0['logweights'] = logweights.masked_fill(
            dropout_mask, float('-inf'))
        return self

    def dropout(self, p):
        new_policy = self.copy()
        return new_policy.dropout_(p)

    def temperature_(self, temp):
        """Apply temperature scaling to mixture logits in place."""
        if isinstance(temp, bool) or not isinstance(temp, (int, float)):
            raise TypeError(f'temperature must be a real scalar, got {type(temp).__name__}.')
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError(f'temperature must be finite and positive, got {temp}.')
        if temp != 1.0:
            logweights = self.denoising_output_x_0['logweights']
            self.denoising_output_x_0['logweights'] = torch.log_softmax(
                logweights / temp, dim=1)
        return self

    def temperature(self, temp):
        new_policy = self.copy()
        return new_policy.temperature_(temp)
