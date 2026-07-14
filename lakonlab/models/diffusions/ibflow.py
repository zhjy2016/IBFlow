# Copyright (c) 2025 Hansheng Chen

import sys
import torch
import torch.nn.functional as F
import mmcv

from copy import deepcopy
from functools import partial
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES

from . import GaussianFlow
from .policies import POLICY_CLASSES, IBFlowPolicy
from lakonlab.utils import module_eval


class IBFlowImitationBase(GaussianFlow):

    def __init__(self, *args, policy_type='IBFlow', policy_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert policy_type in POLICY_CLASSES, \
            f'Invalid policy: {policy_type}. Supported policies are {list(POLICY_CLASSES.keys())}.'
        self.policy_type = policy_type
        self.policy_class = partial(
            POLICY_CLASSES[policy_type], **policy_kwargs
        ) if policy_kwargs else POLICY_CLASSES[policy_type]

    def momentum_integration(
            self,
            sigma_t_src: torch.Tensor, # NFE source time
            x_t_start: torch.Tensor,       # current state
            sigma_t_start: torch.Tensor,   # current integration time
            raw_t_end: torch.Tensor,   # target integration time
            policy,                    # policy predicted at t_src
            eps=1e-4,
            seq_len=None
            ):
        
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        
        sigma_t_end = self.timestep_sampler.warp_t(raw_t_end, seq_len=seq_len)
        sigma_t_end = sigma_t_end.reshape(num_batches, *((ndim - 1) * [1]))

        means = policy.denoising_output_x_0['means_u']
        log_gammas = policy.denoising_output_x_0['loggammas']
        logweights = policy.denoising_output_x_0['logweights']
        
        dt_past = sigma_t_src - sigma_t_start 
        dt_step = sigma_t_start - sigma_t_end   # positive step length

        dt_past = dt_past.unsqueeze(1)
        dt_step = dt_step.unsqueeze(1)
        
        decay_factor = torch.exp(log_gammas * dt_past)
        bs, k, c, h, w = decay_factor.shape
        decay_factor = torch.cat(
            [torch.ones((bs, 1, c, h, w), device=decay_factor.device),
             decay_factor],
            dim=1)
        v_at_a = means * decay_factor
        x_arg = log_gammas * dt_step        
        x_safe = x_arg
        x_sign = torch.sign(x_safe)
        x_sign[x_sign == 0] = 1 
        x_safe = x_sign * torch.clamp(x_safe.abs(), min=eps)
        integral_term = torch.expm1(x_safe) / x_safe
        step_factor = integral_term
        step_factor = torch.cat(
            [torch.ones((bs, 1, c, h, w), device=step_factor.device),
             step_factor],
            dim=1)
        displacement_candidates = v_at_a * dt_step * step_factor        
        weights = torch.softmax(logweights, dim=1)        
        final_displacement = (weights * displacement_candidates).sum(dim=1)
        
        x_t_end = x_t_start - final_displacement
        t_end = sigma_t_end.flatten() * self.num_timesteps
        return x_t_end, sigma_t_end, t_end

    def policy_average_u_momentum(
            self,
            sigma_t_src: torch.Tensor,  # (B, 1, *, 1, 1)
            x_t_start: torch.Tensor,  # (B, C, *, H, W)
            sigma_t_start: torch.Tensor,  # (B, 1, *, 1, 1)
            raw_t_start: torch.Tensor,  # (B, )
            raw_t_end: torch.Tensor,  # (B, )
            total_substeps: int,
            policy,
            seq_len=None,
            eps=1e-4):
        num_batches = x_t_start.size(0)
        ndim = x_t_start.dim()
        is_small_length = torch.round((raw_t_start - raw_t_end) * total_substeps) < 2
        pred_mean_u = pred_local_u = None
        if not is_small_length.all():  # mean velocity over the rollout length
            x_t_end, sigma_t_end, _ = self.momentum_integration(
                sigma_t_src, x_t_start, sigma_t_start, raw_t_end,
                policy, eps=eps, seq_len=seq_len)
            pred_mean_u = (x_t_start - x_t_end) / (sigma_t_start - sigma_t_end).clamp(min=eps)
        if is_small_length.any():  # numerically stable local velocity
            pred_local_u = policy.velocity(sigma_t_src, sigma_t_start)
        if pred_mean_u is None:
            pred_u = pred_local_u
        elif pred_local_u is None:
            pred_u = pred_mean_u
        else:
            pred_u = torch.where(
                is_small_length.reshape(num_batches, *((ndim - 1) * [1])), pred_local_u, pred_mean_u)
        return pred_u

    @staticmethod
    def get_shape_info(x):
        x_t_dst_shape = x.size()
        bs = x_t_dst_shape[0]
        ndim = len(x_t_dst_shape)
        seq_len = x.shape[2:].numel()
        return ndim, bs, seq_len

    def piid_segment_momentum(
            self, teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs, get_x_t_dst=False):
        eps = self.train_cfg.get('eps', 1e-4)
        total_substeps = self.train_cfg.get('total_substeps', 128)
        num_intermediate_states = self.train_cfg.get('num_intermediate_states', 2)
        window_substeps = self.train_cfg.get('window_substeps', 0)

        device = x_t_src.device
        ndim, bs, seq_len = self.get_shape_info(x_t_src)
        if not isinstance(segment_size, torch.Tensor):
            segment_size = torch.tensor(
                [segment_size], dtype=torch.float32, device=device)

        # window size ∆τ ≈ window_substeps / total_substeps
        num_substeps = (segment_size * total_substeps).round().to(torch.long).clamp(min=1)
        substep_size = segment_size / num_substeps
        window_size = torch.minimum(window_substeps * substep_size, segment_size)

        raw_t_dst = raw_t_src - segment_size

        policy_detached = policy.detach()
        if isinstance(policy_detached, IBFlowPolicy):
            gm_dropout = self.train_cfg.get('gm_dropout', 0.0)
            policy_detached.dropout_(gm_dropout)

        # time sampling for scheduled trajectory mixing
        assert not self.timestep_sampler.logit_normal_enable
        student_intervals = torch.rand(
            (bs, num_intermediate_states), device=device
        ) * ((1 - teacher_ratio) * (segment_size - window_size).unsqueeze(-1))
        student_intervals = torch.sort(student_intervals, dim=-1)[0]
        student_intervals = torch.diff(student_intervals, dim=-1, prepend=torch.zeros((bs, 1), device=device))

        teacher_intervals = torch.rand((bs, num_intermediate_states - 1), device=device)
        teacher_intervals = torch.sort(teacher_intervals, dim=-1)[0]
        teacher_intervals = torch.diff(
            teacher_intervals, dim=-1,
            prepend=torch.zeros((bs, 1), device=device),
            append=torch.ones(
                (bs, 1), device=device)
        ) * (teacher_ratio * (segment_size - window_size).unsqueeze(-1))

        x_t = x_t_src
        raw_t = raw_t_src
        sigma_t = sigma_t_src

        all_pred_u = []
        all_tgt_u = []
        all_timesteps = []

        for teacher_step_id in range(num_intermediate_states):
            raw_t_a = (raw_t - student_intervals[:, teacher_step_id]).clamp(min=0)
            raw_t_b = (raw_t_a - teacher_intervals[:, teacher_step_id]).clamp(min=0)

            with torch.no_grad(), module_eval(teacher):
                x_t_a, sigma_t_a, t_a = self.momentum_integration(
                    sigma_t_src, x_t, sigma_t, raw_t_a, 
                    policy_detached, eps=eps, seq_len=seq_len)
                tgt_u = teacher(return_u=True, x_t=x_t_a, t=t_a, **teacher_kwargs)
                all_tgt_u.append(tgt_u)
                all_timesteps.append(t_a)

            pred_u = self.policy_average_u_momentum(
                sigma_t_src, 
                x_t_a, sigma_t_a, raw_t_a, raw_t_b - window_size, total_substeps,
                policy, seq_len=seq_len, eps=eps)
            all_pred_u.append(pred_u)

            sigma_t_b = self.timestep_sampler.warp_t(raw_t_b, seq_len=seq_len).reshape(bs, *((ndim - 1) * [1]))
            x_t = x_t_a + tgt_u * (sigma_t_b - sigma_t_a)
            raw_t = raw_t_b
            sigma_t = sigma_t_b

        loss_kwargs = dict(
            u_t_pred=torch.cat(all_pred_u, dim=0),
            u_t=torch.cat(all_tgt_u, dim=0),
            timesteps=torch.cat(all_timesteps, dim=0)
        )
        loss = self.flow_loss(loss_kwargs)

        if get_x_t_dst:
            with torch.no_grad():
                x_t_dst, _, _ = self.momentum_integration(
                    sigma_t_src, x_t, sigma_t, raw_t_dst,
                    policy_detached, eps=eps, seq_len=seq_len)
        else:
            x_t_dst = None

        return loss, x_t_dst, raw_t_dst
    
    def forward_test(
            self, x_0=None, noise=None, guidance_scale=None,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t_src = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()  # h * w or t * h * w
        ori_dtype = x_t_src.dtype
        device = x_t_src.device
        x_t_src = x_t_src.float()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        eps = cfg.get('eps', 1e-4)
        nfe = cfg['nfe']
        timestep_ratio = max(cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)

        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        if show_pbar:
            pbar = mmcv.ProgressBar(self.distill_steps)

        # ========== Main sampling loop ==========
        for step_id in range(nfe):
            is_final_step = step_id == nfe - 1
            if is_final_step:
                segment_size = base_segment_size * timestep_ratio
            else:
                segment_size = base_segment_size

            raw_t_dst = raw_t_src - segment_size

            denoising_output = self.pred(x_t_src, t_src, **kwargs)
            policy = self.policy_class(
                denoising_output, x_t_src, sigma_t_src, eps=eps)
            if isinstance(policy, IBFlowPolicy) and not is_final_step:
                temperature = cfg.get('temperature', 1.0)
                policy.temperature_(temperature)

            x_t_dst, sigma_t_dst, t_dst = self.momentum_integration(
                    sigma_t_src, x_t_src, sigma_t_src, raw_t_dst, 
                    policy, eps=1e-4, seq_len=seq_len)

            x_t_src = x_t_dst
            raw_t_src = raw_t_dst
            sigma_t_src = sigma_t_dst
            t_src = t_dst

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t_src.to(ori_dtype)


@MODULES.register_module()
class IBFlowImitationDataFree(IBFlowImitationBase):

    is_multistep = True

    def forward_initialize(
            self, x_0, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        device = get_module_device(self)
        num_batches = x_0.size(0)  # x_0 is a dummy input

        num_decay_iters = self.train_cfg.get('num_decay_iters', 0)
        if num_decay_iters > 0:
            teacher_ratio = 1 - min(running_status['iteration'], num_decay_iters) / num_decay_iters
            log_vars = dict(teacher_ratio=teacher_ratio)
        else:
            teacher_ratio = 0.0
            log_vars = dict()

        x_t_src = torch.randn_like(x_0)
        raw_t_src = torch.ones((num_batches,), dtype=torch.float32, device=device)
        step_states = dict(
            step_id=0,
            terminate=False,
            detachable=True,
            teacher_ratio=teacher_ratio,
            x_t_src=x_t_src,
            raw_t_src=raw_t_src,
        )

        return step_states, log_vars

    def forward_train(
            self, x_0, step_states=None, teacher=None, teacher_kwargs=dict(), running_status=None, **kwargs):
        step_id = step_states['step_id']
        teacher_ratio = step_states['teacher_ratio']
        x_t_src = step_states['x_t_src']
        raw_t_src = step_states['raw_t_src']

        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ndim = x_t_src.dim()
        assert ndim in [4, 5], f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.'

        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        timestep_ratio = max(self.train_cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)
        is_final_step = step_id == nfe - 1
        if is_final_step:
            segment_size = base_segment_size * timestep_ratio
        else:
            segment_size = base_segment_size

        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(denoising_output, x_t_src, sigma_t_src)

        step_loss_diffusion, x_t_dst, raw_t_dst = self.piid_segment_momentum(
            teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs, get_x_t_dst=True)

        # print(f"Step {step_id}: loss_diffusion = {step_loss_diffusion.item():.6f}")
        loss_diffusion = step_loss_diffusion * segment_size  # Weighing by segment size
        loss = loss_diffusion

        log_vars = {k: v * segment_size for k, v in self.flow_loss.log_vars.items()}
        log_vars.update({
            'loss_diffusion': float(loss_diffusion),
            f'loss_diffusion_step{step_id}': float(step_loss_diffusion),
        })

        if step_id < nfe - 1:
            step_states.update(
                step_id=step_id + 1,
                x_t_src=x_t_dst,
                raw_t_src=raw_t_dst)
        else:
            step_states.update(terminate=True)

        return loss, log_vars, step_states

    def forward(self, x_0=None, return_step_states=False, **kwargs):
        if return_step_states:
            return self.forward_initialize(x_0=x_0, **kwargs)
        else:
            return super().forward(x_0=x_0, **kwargs)


@MODULES.register_module()
class IBFlowImitationDataFreeLinearDynamicCFG(IBFlowImitationDataFree):
    """Legacy-compatible recipe used by the 2-NFE 88.67 release candidate.

    This class preserves the exact training objective recorded by the original
    ``ArcFlowImitationDataFreeDynamicCFG`` experiment while using the public
    IBFlow architecture and policy names. It is intentionally separate from
    :class:`IBFlowImitationDataFreeIBCFG`: the release recipe uses a random
    look-ahead target and a linear guidance schedule, whereas IBCFG uses the
    information-bottleneck target and SNR schedule described by the paper.

    Args:
        cfg_scale_tau (float): Base scale for look-ahead CFG supervision.
        lambda_cfg_loss (float): Weight of the auxiliary CFG loss.
        scale_alpha_eq (float): Linear multiplier for the segment teacher.
        scale_alpha_gt (float): Linear multiplier for look-ahead CFG.
    """

    def __init__(self,
                 *args,
                 cfg_scale_tau=1.0,
                 lambda_cfg_loss=1.0,
                 scale_alpha_eq=0.0,
                 scale_alpha_gt=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_scale_tau = cfg_scale_tau
        self.lambda_cfg_loss = lambda_cfg_loss
        self.scale_alpha_eq = scale_alpha_eq
        self.scale_alpha_gt = scale_alpha_gt
        self.eps_cfg = 1e-5

    def forward_train(
            self,
            x_0,
            step_states=None,
            teacher=None,
            teacher_kwargs=None,
            uncond_teacher_kwargs=None,
            running_status=None,
            **kwargs):
        teacher_kwargs = {} if teacher_kwargs is None else teacher_kwargs
        device = get_module_device(self)
        step_id = step_states['step_id']
        teacher_ratio = step_states['teacher_ratio']
        x_t_src = step_states['x_t_src']
        raw_t_src = step_states['raw_t_src']

        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ndim = x_t_src.dim()
        if ndim not in (4, 5):
            raise ValueError(
                f'Invalid x_t_src shape: {x_t_src.shape}. Expected 4D or 5D tensor.')

        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        timestep_ratio = max(self.train_cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)
        is_final_step = step_id == nfe - 1
        segment_size = (
            base_segment_size * timestep_ratio if is_final_step
            else base_segment_size)

        sigma_t_src = self.timestep_sampler.warp_t(
            raw_t_src, seq_len=seq_len).reshape(
                num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(
            denoising_output, x_t_src, sigma_t_src)

        if not torch.allclose(raw_t_src, raw_t_src[0]):
            raise ValueError(
                'Linear Dynamic-CFG expects a synchronized data-free timestep '
                'within each batch.')
        t_value = raw_t_src[0].item()

        dynamic_teacher_kwargs = deepcopy(teacher_kwargs)
        if 'guidance_scale' in dynamic_teacher_kwargs:
            base_guidance_scale = dynamic_teacher_kwargs['guidance_scale']
            multiplier = 1.0 + self.scale_alpha_eq * t_value
            if isinstance(base_guidance_scale, (list, tuple)):
                dynamic_guidance_scale = type(base_guidance_scale)(
                    value * multiplier for value in base_guidance_scale)
            else:
                dynamic_guidance_scale = base_guidance_scale * multiplier
            dynamic_teacher_kwargs['guidance_scale'] = dynamic_guidance_scale

        step_loss_diffusion, x_t_dst, raw_t_dst = self.piid_segment_momentum(
            teacher,
            policy,
            x_t_src,
            raw_t_src,
            sigma_t_src,
            teacher_ratio,
            segment_size,
            dynamic_teacher_kwargs,
            get_x_t_dst=True)
        loss_diffusion = step_loss_diffusion * segment_size

        loss_cfg = torch.zeros((), device=x_0.device)
        if self.lambda_cfg_loss > 0 and uncond_teacher_kwargs is not None:
            weights = torch.softmax(denoising_output['logweights'], dim=1)
            student_v_pred = (
                weights * denoising_output['means']).sum(dim=1)

            with torch.no_grad():
                lookahead_ratio = torch.rand(
                    (num_batches,), device=device).clamp(min=0.1, max=0.8)
                raw_t_ca = raw_t_src * lookahead_ratio
                sigma_t_ca = self.timestep_sampler.warp_t(
                    raw_t_ca, seq_len=seq_len).reshape(
                        num_batches, *((ndim - 1) * [1]))
                t_ca = sigma_t_ca.flatten() * self.num_timesteps

                x_t_ca, _, _ = self.momentum_integration(
                    sigma_t_src,
                    x_t_src,
                    sigma_t_src,
                    raw_t_ca,
                    policy,
                    eps=self.eps_cfg,
                    seq_len=seq_len)

                v_teacher_cond = teacher(
                    return_u=True,
                    x_t=x_t_ca,
                    t=t_ca,
                    **teacher_kwargs)
                v_teacher_uncond = teacher(
                    return_u=True,
                    x_t=x_t_ca,
                    t=t_ca,
                    **uncond_teacher_kwargs)

                dynamic_scale_tau = self.cfg_scale_tau * (
                    1.0 + self.scale_alpha_gt * t_value)
                cfg_update = dynamic_scale_tau * (
                    v_teacher_cond - v_teacher_uncond)

            target_v_ca = (student_v_pred + cfg_update).detach()
            raw_loss_cfg = F.mse_loss(student_v_pred, target_v_ca)
            loss_cfg = raw_loss_cfg * segment_size

        loss_total = loss_diffusion + self.lambda_cfg_loss * loss_cfg
        log_vars = {
            key: value * segment_size
            for key, value in self.flow_loss.log_vars.items()
        }
        log_vars.update({
            'loss_diffusion': float(loss_diffusion),
            f'loss_diffusion_step{step_id}': float(step_loss_diffusion),
            f'loss_cfg_step{step_id}': float(loss_cfg),
            'loss_cfg': float(loss_cfg),
            'loss_total': float(loss_total),
        })

        if step_id < nfe - 1:
            step_states.update(
                step_id=step_id + 1,
                x_t_src=x_t_dst,
                raw_t_src=raw_t_dst)
        else:
            step_states.update(terminate=True)

        return loss_total, log_vars, step_states


@MODULES.register_module()
class IBFlowImitationDataFreeIBCFG(IBFlowImitationDataFree):
    """IBFlow data-free distillation + Information Bottleneck-guided dynamic CFG.

    Dual-track framework following Eq.(4):
        ∆θ ∝ ∆_DM(τ_DM) + λ · ∆_CA(τ*_CA, ω*(t))
             Shield            Spear

    Shield: Original IBFlow PIID segment matching with teacher's default CFG.
    Spear:  Dynamic CA with IB-optimal τ*_CA and SNR-driven ω*(t).

    Hyperparameters (paper Section 4.1):
        kappa (κ):         IB divergence budget √(2δ), controls injection stride.
                           Table 4 ablation: {0.5, 1.0, 1.5, 3.0}, best=1.5
        omega_max:         Maximum guidance scale at t→noise end.
                           Inherited from teacher's default CFG scale (e.g. 4.0).
        gamma_snr (γ):     Temperature for SNR sensitivity. Default=1.0.
        lambda_cfg_loss:   Weight of the CA spear loss relative to Shield.
    """

    def __init__(self, *args,
                 lambda_cfg_loss=1.0,
                 kappa=1.5,
                 omega_max=4.0,
                 gamma_snr=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cfg_loss = lambda_cfg_loss
        self.kappa = kappa
        self.omega_max = omega_max
        self.gamma_snr = gamma_snr
        self.eps_cfg = 1e-5


    # -----------------------------------------------------------------
    # Theorem 2, Eq.(13): Entropy-aware dynamic injection strength
    # -----------------------------------------------------------------
    def _compute_omega_star(self, raw_t):
        """ω*(t) = 1 + (ω_max - 1) / (1 + γ · SNR(t))

        Coordinate mapping (IBFlow implementation ↔ paper):
            IBFlow: raw_t=1 → pure noise,  raw_t=0 → clean image
            Paper:   t=0     → pure noise,  t=1     → clean image
            Relation: paper_t = 1 - raw_t

        Paper's SNR(t) = t² / (1-t)² with paper t.
        In IBFlow coordinates: SNR = (1 - raw_t)² / raw_t²
          - raw_t → 1 (noise): SNR → 0, ω* → ω_max  (maximum guidance)
          - raw_t → 0 (clean): SNR → ∞, ω* → 1      (no extra guidance)
        """
        snr_t = ((1.0 - raw_t) ** 2) / (raw_t ** 2 + self.eps_cfg)
        omega_t = 1.0 + (self.omega_max - 1.0) / (1.0 + self.gamma_snr * snr_t)
        return omega_t

    # -----------------------------------------------------------------
    # Theorem 1, Eq.(11): Instance-aware dynamic injection target
    # -----------------------------------------------------------------
    def _compute_tau_star(self, raw_t, delta_v_norm):
        """τ*_CA = min(1, t + κ / (‖Δv_t‖₂ + ε))   [paper coords]

        In IBFlow coordinates (raw_t decreases toward clean):
          raw_t_CA = max(min_t, raw_t - κ / (‖Δv_t‖₂ + ε))

        Clamped to min=0.02 to prevent momentum_integration from
        extrapolating to t=0 where the teacher's denoiser becomes
        numerically degenerate. Each sample gets its own τ*_CA.
        """
        tau_offset = self.kappa / (delta_v_norm + self.eps_cfg)
        raw_t_CA = torch.clamp(raw_t - tau_offset, min=0.02)
        return raw_t_CA

    # -----------------------------------------------------------------
    # Helpers: extract cond / uncond embeddings from teacher_kwargs
    # -----------------------------------------------------------------
    def _extract_cond_kwargs(self, teacher_kwargs, kwargs):
        """Extract pure conditional prompt embeddings.

        teacher_kwargs typically contains [neg, pos] concatenated embeddings
        + guidance_scale > 1.0. We chunk the second half (pos) to get pure
        conditional embeddings for calling teacher at guidance_scale=1.0.
        """
        guidance_scale = teacher_kwargs.get('guidance_scale', 1.0)
        has_cfg_concat = (isinstance(guidance_scale, (int, float)) and guidance_scale > 1.0)

        if has_cfg_concat:
            cond_kwargs = {}
            skip_keys = {'guidance_scale', 'guidance', 'test_cfg_override'}
            for k, v in teacher_kwargs.items():
                if k in skip_keys:
                    continue
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    half = v.size(0) // 2
                    cond_kwargs[k] = v[half:]
                else:
                    cond_kwargs[k] = v
            return cond_kwargs
        else:
            return {k: v for k, v in teacher_kwargs.items()
                    if k not in {'guidance_scale', 'guidance', 'test_cfg_override'}}

    def _extract_uncond_kwargs(self, teacher_kwargs, uncond_teacher_kwargs):
        """Extract pure unconditional prompt embeddings.

        If uncond_teacher_kwargs is provided from upper layer, use directly.
        Otherwise, chunk first half of teacher_kwargs ([neg, pos] concat).
        """
        if uncond_teacher_kwargs is not None:
            return {k: v for k, v in uncond_teacher_kwargs.items()
                    if k not in {'guidance_scale', 'guidance', 'test_cfg_override'}}

        guidance_scale = teacher_kwargs.get('guidance_scale', 1.0)
        has_cfg_concat = (isinstance(guidance_scale, (int, float)) and guidance_scale > 1.0)

        if has_cfg_concat:
            uncond_kwargs = {}
            skip_keys = {'guidance_scale', 'guidance', 'test_cfg_override'}
            for k, v in teacher_kwargs.items():
                if k in skip_keys:
                    continue
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    half = v.size(0) // 2
                    uncond_kwargs[k] = v[:half]
                else:
                    uncond_kwargs[k] = v
            return uncond_kwargs
        else:
            raise ValueError(
                "Cannot extract uncond embeddings: teacher_kwargs has no CFG concat "
                "and uncond_teacher_kwargs was not provided.")

    # -----------------------------------------------------------------
    # Main training forward
    # -----------------------------------------------------------------
    def forward_train(self, x_0, step_states=None, teacher=None,
                      teacher_kwargs=dict(), uncond_teacher_kwargs=None,
                      running_status=None, **kwargs):
        """Forward training with IB-guided dynamic CFG distillation.

        Eq.(4): ∆θ ∝ [∆_DM(τ_DM) + λ · ∆_CA(τ*_CA, ω*(t))] · ∂x_t/∂θ
                       Shield         Spear
        """
        # =================================================================
        # Step A: State unpacking and timestep computation (IBFlow)
        # =================================================================
        device = get_module_device(self)
        step_id = step_states['step_id']
        teacher_ratio = step_states['teacher_ratio']
        x_t_src = step_states['x_t_src']
        raw_t_src = step_states['raw_t_src']

        num_batches = x_t_src.size(0)
        seq_len = x_t_src.shape[2:].numel()
        ndim = x_t_src.dim()

        eps = self.train_cfg.get('eps', 1e-4)
        nfe = self.train_cfg['nfe']
        timestep_ratio = max(self.train_cfg.get('timestep_ratio', 1.0), eps)
        base_segment_size = 1 / (nfe - 1 + timestep_ratio)
        is_final_step = step_id == nfe - 1
        segment_size = base_segment_size * timestep_ratio if is_final_step else base_segment_size

        sigma_t_src = self.timestep_sampler.warp_t(raw_t_src, seq_len=seq_len).reshape(
            num_batches, *((ndim - 1) * [1]))
        t_src = sigma_t_src.flatten() * self.num_timesteps

        # =================================================================
        # Step B: Student forward (single NFE → Gaussian mixture params)
        # =================================================================
        denoising_output = self.pred(x_t_src, t_src, **kwargs)
        policy = self.policy_class(denoising_output, x_t_src, sigma_t_src)

        # =================================================================
        # Step C: Shield — DM Regularizer (IBFlow PIID)
        #   Uses teacher_kwargs with the teacher CFG scale.
        #   Corresponds to ∆_DM(τ_DM) in Eq.(4).
        # =================================================================
        step_loss_diffusion, x_t_dst, raw_t_dst = self.piid_segment_momentum(
            teacher, policy, x_t_src, raw_t_src, sigma_t_src, teacher_ratio, segment_size,
            teacher_kwargs, get_x_t_dst=True)
        loss_diffusion = step_loss_diffusion * segment_size

        # Save Shield's log_vars BEFORE Spear calls self.flow_loss,
        # which would overwrite self.flow_loss.log_vars
        shield_log_vars = dict(self.flow_loss.log_vars)

        # =================================================================
        # Step D: Spear — CA Engine (IB-guided Dynamic CFG Augmentation)
        #   Eq.(14): ∆_CA ∝ (ω*(t)-1) · [v^c_{τ*}(x_{τ*}) - v^u_{τ*}(x_{τ*})]
        #
        #   Implementation: velocity matching (NOT SDS-style).
        #   Student velocity at τ*_CA → match → teacher dynamic CFG target at τ*_CA.
        #   Uses self.flow_loss (same DiffusionMSELoss as Shield) for magnitude parity.
        # =================================================================
        loss_cfg = torch.tensor(0.0, device=device)

        if self.lambda_cfg_loss > 0:
            with torch.no_grad(), module_eval(teacher):
                # --- D-1: Construct pure cond / uncond kwargs ---
                cond_kw = self._extract_cond_kwargs(teacher_kwargs, kwargs)
                uncond_kw = self._extract_uncond_kwargs(teacher_kwargs, uncond_teacher_kwargs)

                # --- D-2: Theorem 1 — Instance-aware τ*_CA ---
                # Query teacher at current t for cond/uncond velocities
                v_tea_cond_t = teacher(return_u=True, x_t=x_t_src, t=t_src, **cond_kw)
                v_tea_uncond_t = teacher(return_u=True, x_t=x_t_src, t=t_src, **uncond_kw)

                # ‖Δv_t(x_t)‖₂ per sample, shape: (B,)
                delta_v_t = v_tea_cond_t - v_tea_uncond_t
                delta_v_norm = torch.norm(
                    delta_v_t.reshape(num_batches, -1), p=2, dim=1)

                # Eq.(11): τ*_CA in IBFlow coordinates
                raw_t_CA = self._compute_tau_star(raw_t_src, delta_v_norm)

                # --- D-3: Roll student policy forward to τ*_CA ---
                policy_detached = policy.detach()
                sigma_t_CA = self.timestep_sampler.warp_t(
                    raw_t_CA, seq_len=seq_len).reshape(
                    num_batches, *((ndim - 1) * [1]))
                t_CA = sigma_t_CA.flatten() * self.num_timesteps

                x_t_CA, _, _ = self.momentum_integration(
                    sigma_t_src, x_t_src, sigma_t_src, raw_t_CA,
                    policy_detached, eps=eps, seq_len=seq_len)

                # --- D-4: Teacher velocities at τ*_CA ---
                v_tea_cond_CA = teacher(
                    return_u=True, x_t=x_t_CA, t=t_CA, **cond_kw)
                v_tea_uncond_CA = teacher(
                    return_u=True, x_t=x_t_CA, t=t_CA, **uncond_kw)

                # --- D-5: Theorem 2, Eq.(13) — SNR-driven ω*(t) ---
                omega_t = self._compute_omega_star(raw_t_src)
                omega_t_bc = omega_t.reshape(num_batches, *((ndim - 1) * [1]))

                # --- D-6: Eq.(14) — Dynamic CFG target velocity ---
                # tgt = v^u + ω*(t) · (v^c - v^u) at τ*_CA
                tgt_u_CA = v_tea_uncond_CA + omega_t_bc * (v_tea_cond_CA - v_tea_uncond_CA)

            # --- D-7: Student velocity at τ*_CA (WITH gradient) ---
            # policy_average_u_momentum with start=end → is_small_length=True
            # → calls policy.velocity(sigma_t_src, sigma_t_CA), which correctly
            #   applies decay_factor from loggammas. This is NOT the same as
            #   raw softmax(logweights) * means!
            pred_u_CA = self.policy_average_u_momentum(
                sigma_t_src,
                x_t_CA, sigma_t_CA, raw_t_CA, raw_t_CA,
                self.train_cfg.get('total_substeps', 128),
                policy, seq_len=seq_len, eps=eps)

            # --- D-8: Velocity matching loss via self.flow_loss ---
            # This ensures the Spear loss goes through exactly the same
            # pipeline as Shield: flatmean → *0.5 → *scale(30.0) → mean,
            # so their gradient magnitudes are naturally balanced.
            loss_kwargs_cfg = dict(
                u_t_pred=pred_u_CA,
                u_t=tgt_u_CA,
                timesteps=t_CA,
            )
            raw_loss_cfg = self.flow_loss(loss_kwargs_cfg)
            loss_cfg = raw_loss_cfg * segment_size

        # =================================================================
        # Step E: Total loss & state update
        # =================================================================
        loss_total = loss_diffusion + self.lambda_cfg_loss * loss_cfg

        log_vars = {k: v * segment_size for k, v in shield_log_vars.items()}
        log_vars.update({
            'loss_diffusion': float(loss_diffusion),
            f'loss_diffusion_step{step_id}': float(step_loss_diffusion),
            f'loss_cfg_step{step_id}': float(loss_cfg),
            'loss_cfg': float(loss_cfg),
            'loss_total': float(loss_total),
        })

        if step_id < nfe - 1:
            step_states.update(
                step_id=step_id + 1,
                x_t_src=x_t_dst,
                raw_t_src=raw_t_dst)
        else:
            step_states.update(terminate=True)

        return loss_total, log_vars, step_states
