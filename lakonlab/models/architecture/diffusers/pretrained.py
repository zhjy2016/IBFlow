import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKLQwenImage
from diffusers.pipelines import QwenImagePipeline
from mmgen.models.builder import MODULES

for name in (
        'transformers.tokenization_utils_base',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_fast'):
    logging.getLogger(name).setLevel(logging.ERROR)

@MODULES.register_module()
class PretrainedVAEQwenImage(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 use_slicing=False,
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            from_pretrained, **kwargs)
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, *args, **kwargs):
        return self.vae(*args, return_dict=False, **kwargs)[0]

    def encode(self, img):
        device = img.device
        dtype = img.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return ((self.vae.encode(img.unsqueeze(-3)).latent_dist.sample() - latents_mean) / latents_std).squeeze(-3)

    def decode(self, code):
        device = code.device
        dtype = code.dtype
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1)
        return self.vae.decode(code.unsqueeze(-3) * latents_std + latents_mean, return_dict=False)[0].squeeze(-3)


@MODULES.register_module()
class PretrainedQwenImageTextEncoder(nn.Module):
    def __init__(self,
                 from_pretrained='Qwen/Qwen-Image',
                 freeze=True,
                 eval_mode=True,
                 torch_dtype='bfloat16',
                 max_sequence_length=512,
                 pad_seq_len=None,
                 **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        if pad_seq_len is not None:
            assert pad_seq_len >= max_sequence_length
        self.pad_seq_len = pad_seq_len
        self.pipeline = QwenImagePipeline.from_pretrained(
            from_pretrained,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=getattr(torch, torch_dtype),
            **kwargs)
        self.text_encoder = self.pipeline.text_encoder
        self.freeze = freeze
        self.eval_mode = eval_mode
        if self.freeze:
            self.requires_grad_(False)
        if self.eval_mode:
            self.eval()

    def train(self, mode=True):
        mode = mode and (not self.eval_mode)
        return super().train(mode)

    def forward(self, prompt):
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, max_sequence_length=self.max_sequence_length)
        if self.pad_seq_len is not None:
            pad_len = self.pad_seq_len - prompt_embeds.size(1)
            prompt_embeds = F.pad(
                prompt_embeds, (0, 0, 0, pad_len), value=0.0)
            prompt_embeds_mask = F.pad(
                prompt_embeds_mask, (0, pad_len), value=0.0)
        return dict(
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask)
