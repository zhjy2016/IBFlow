import argparse
import hashlib
import json
import os
from collections import Counter, OrderedDict

from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from mmcv import Config
from mmgen.models import build_module
from safetensors.torch import save_file

from lakonlab import models  # noqa: F401
from lakonlab.runner.checkpoint import _load_checkpoint


CONFIG_ALIASES = {
    'ArcFlow': 'IBFlow',
    'ArcQwenImageTransformer2DModel': 'IBQwenImageTransformer2DModel',
    'ArcFlowImitationDataFreeDynamicCFG':
        'IBFlowImitationDataFreeLinearDynamicCFG',
    'ArcFlowImitationDataFreeIBCFG': 'IBFlowImitationDataFreeIBCFG',
}

DIFFUSION_SIGNATURE_KEYS = (
    'cfg_scale_tau',
    'lambda_cfg_loss',
    'scale_alpha_eq',
    'scale_alpha_gt',
    'kappa',
    'omega_max',
    'gamma_snr',
    'policy_type',
)

DENOISER_SIGNATURE_KEYS = (
    'type',
    'num_gaussians',
    'logweights_channels',
    'in_channels',
    'out_channels',
    'num_layers',
    'attention_head_dim',
    'num_attention_heads',
    'joint_attention_dim',
    'axes_dims_rope',
    'use_lora',
    'lora_rank',
    'lora_target_modules',
)

TRAIN_SIGNATURE_KEYS = (
    'num_intermediate_states',
    'teacher_guidance_scale',
    'nfe',
    'timestep_ratio',
    'total_substeps',
    'num_decay_iters',
)

REQUIRED_OUTPUT_KEYS = {
    'norm_out.linear.bias',
    'norm_out.linear.weight',
    'proj_out_loggamma.bias',
    'proj_out_loggamma.weight',
    'proj_out_logweights.bias',
    'proj_out_logweights.weight',
    'proj_out_means.bias',
    'proj_out_means.weight',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a verified IBFlow checkpoint for diffusers inference.')
    parser.add_argument('config', help='Exact training config used for the checkpoint.')
    parser.add_argument('--ckpt', required=True, help='Path to the .pth checkpoint.')
    parser.add_argument(
        '--out-dir', required=True,
        help='Directory for config, safetensors weights, and manifest.')
    parser.add_argument('--base-model', default='Qwen/Qwen-Image')
    parser.add_argument(
        '--base-revision', default=None,
        help='Immutable upstream base-model revision recorded in the manifest.')
    parser.add_argument(
        '--base-revision-source', default=None,
        help='Provenance namespace for the recorded base revision IDs.')
    parser.add_argument(
        '--base-transformer-revision', default=None,
        help=(
            'Optional transformer-shard revision for snapshots assembled '
            'from more than one upstream revision.'))
    parser.add_argument(
        '--non-ema', action='store_true',
        help='Export non-EMA weights instead of EMA weights.')
    parser.add_argument(
        '--allow-config-mismatch', action='store_true',
        help='Export despite missing/mismatched checkpoint config metadata; the manifest is marked unverified.')
    return parser.parse_args()


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(value):
    if isinstance(value, dict):
        return {
            key: canonicalize(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, str):
        return CONFIG_ALIASES.get(value, value)
    return value


def config_signature(cfg):
    diffusion = cfg.model.diffusion
    denoiser = diffusion.denoising
    signature = {
        'diffusion_type': diffusion.type,
        'diffusion': {
            key: diffusion[key]
            for key in DIFFUSION_SIGNATURE_KEYS if key in diffusion
        },
        'denoiser': {
            key: denoiser[key]
            for key in DENOISER_SIGNATURE_KEYS if key in denoiser
        },
        'train_cfg': {
            key: cfg.train_cfg[key]
            for key in TRAIN_SIGNATURE_KEYS if key in cfg.train_cfg
        },
        'test_cfg': {
            key: cfg.test_cfg[key]
            for key in ('nfe', 'timestep_ratio', 'total_substeps')
            if key in cfg.test_cfg
        },
        'scheduler_shift': diffusion.timestep_sampler.get('shift'),
    }
    return canonicalize(signature)


def signature_sha256(signature):
    payload = json.dumps(
        signature, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def checkpoint_config(checkpoint):
    config_text = checkpoint.get('meta', {}).get('config')
    if not config_text:
        return None
    return Config.fromstring(config_text, '.py')


def validate_export_tensors(tensors):
    keys = set(tensors)
    missing = REQUIRED_OUTPUT_KEYS - keys
    if missing:
        raise ValueError(f'Checkpoint is missing output tensors: {sorted(missing)}.')

    lora_keys = {key for key in keys if 'lora_' in key}
    lora_a = {
        key.removesuffix('.lora_A.weight')
        for key in lora_keys if key.endswith('.lora_A.weight')
    }
    lora_b = {
        key.removesuffix('.lora_B.weight')
        for key in lora_keys if key.endswith('.lora_B.weight')
    }
    if not lora_a or lora_a != lora_b:
        raise ValueError(
            'LoRA tensors are missing or unpaired: '
            f'A-only={sorted(lora_a - lora_b)}, '
            f'B-only={sorted(lora_b - lora_a)}.')


def save_config(model, save_directory, class_name_override):
    if os.path.isfile(save_directory):
        raise ValueError(f'Output path is a file: {save_directory}')
    os.makedirs(save_directory, exist_ok=True)

    config_dict = json.loads(model.to_json_string())
    config_dict['_class_name'] = class_name_override
    output_config_file = os.path.join(save_directory, model.config_name)
    with open(output_config_file, 'w', encoding='utf-8') as writer:
        json.dump(config_dict, writer, indent=2, sort_keys=True)
        writer.write('\n')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    checkpoint = _load_checkpoint(args.ckpt, map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise KeyError(f'Checkpoint has no state_dict: {args.ckpt}')

    requested_signature = config_signature(cfg)
    source_cfg = checkpoint_config(checkpoint)
    source_signature = None if source_cfg is None else config_signature(source_cfg)
    config_verified = source_signature == requested_signature
    if not config_verified and not args.allow_config_mismatch:
        if source_signature is None:
            raise ValueError(
                'Checkpoint has no meta.config; pass --allow-config-mismatch '
                'only for a documented legacy export.')
        raise ValueError(
            'The export config does not match the checkpoint training signature. '
            'Use the exact dumped config or pass --allow-config-mismatch to create '
            'an explicitly unverified artifact.')

    prefix = (
        'diffusion.denoising.' if args.non_ema
        else 'diffusion_ema.denoising.')
    out_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if not key.startswith(prefix):
            continue
        key = key[len(prefix):]
        key = key.replace('lora_A.default.weight', 'lora_A.weight')
        key = key.replace('lora_B.default.weight', 'lora_B.weight')
        if key in out_dict:
            raise ValueError(f'Duplicate exported tensor key: {key}.')
        out_dict[key] = value
    if not out_dict:
        raise ValueError(f'No weights found with checkpoint prefix {prefix!r}.')
    validate_export_tensors(out_dict)

    denoising_cfg = cfg.model.diffusion.denoising.copy()
    denoising_cfg.use_lora = False
    denoising_cfg.pretrained = None
    denoising_cfg.freeze_exclude = None
    model = build_module(denoising_cfg)
    save_config(model, args.out_dir, denoising_cfg.type)

    policy_config = dict(cfg.model.diffusion.get('policy_kwargs', {}))
    policy_config['type'] = cfg.model.diffusion.get('policy_type', 'IBFlow')
    checkpoint_iteration = checkpoint.get('meta', {}).get('iter')
    metadata = {
        'format_version': '1',
        'policy_config': json.dumps(policy_config, sort_keys=True),
        'base_model': args.base_model,
        'base_revision': args.base_revision or '',
        'base_transformer_revision': args.base_transformer_revision or '',
        'base_revision_source': args.base_revision_source or '',
        'training_class': str(requested_signature['diffusion_type']),
        'config_signature_sha256': signature_sha256(requested_signature),
        'config_verified': json.dumps(config_verified),
        'checkpoint_iteration': str(checkpoint_iteration or ''),
        'weight_source': 'online' if args.non_ema else 'ema',
    }
    weights_path = os.path.join(args.out_dir, SAFETENSORS_WEIGHTS_NAME)
    save_file(out_dict, weights_path, metadata=metadata)

    dtype_counts = Counter(str(tensor.dtype) for tensor in out_dict.values())
    manifest = {
        'format_version': 1,
        'model_type': 'IBFlow-Qwen',
        'base_model': args.base_model,
        'base_revision': args.base_revision,
        'base_transformer_revision': args.base_transformer_revision,
        'base_revision_source': args.base_revision_source,
        'adapter': {
            'filename': SAFETENSORS_WEIGHTS_NAME,
            'sha256': sha256_file(weights_path),
            'size_bytes': os.path.getsize(weights_path),
            'tensor_count': len(out_dict),
            'dtype_counts': dict(sorted(dtype_counts.items())),
            'weight_source': 'online' if args.non_ema else 'ema',
        },
        'training': {
            'class': requested_signature['diffusion_type'],
            'checkpoint_iteration': checkpoint_iteration,
            'config_filename': os.path.basename(args.config),
            'config_signature': requested_signature,
            'config_signature_sha256': signature_sha256(requested_signature),
            'config_verified_against_checkpoint': config_verified,
            'source_checkpoint_filename': os.path.basename(args.ckpt),
            'source_checkpoint_sha256': sha256_file(args.ckpt),
        },
        'inference_defaults': {
            'nfe': cfg.test_cfg.get('nfe'),
            'timestep_ratio': cfg.test_cfg.get('timestep_ratio'),
            'scheduler_shift': cfg.model.diffusion.timestep_sampler.get('shift'),
            'height': 1024,
            'width': 1024,
            'dtype': 'bfloat16',
        },
    }
    manifest_path = os.path.join(args.out_dir, 'ibflow_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as writer:
        json.dump(manifest, writer, indent=2, sort_keys=True)
        writer.write('\n')

    print(
        f'Exported {len(out_dict)} tensors to {weights_path}; '
        f'manifest: {manifest_path}')


if __name__ == '__main__':
    main()
