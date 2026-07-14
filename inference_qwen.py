import argparse
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from lakonlab.pipelines import IBQwenImagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Generate an image with IBFlow-Qwen.')
    parser.add_argument('--prompt', required=True, help='Text prompt.')
    parser.add_argument('--output', default='ibflow_qwen.png', help='Output image path.')
    parser.add_argument('--base-model', default='Qwen/Qwen-Image', help='Qwen-Image model ID or local path.')
    parser.add_argument('--base-revision', default=None, help='Optional immutable base-model revision.')
    parser.add_argument('--adapter', required=True, help='IBFlow adapter repository or local directory.')
    parser.add_argument('--adapter-revision', default=None, help='Optional immutable adapter revision.')
    parser.add_argument('--adapter-subfolder', default=None, help='Optional adapter subfolder.')
    parser.add_argument('--nfe', type=int, default=2, help='Number of student function evaluations.')
    parser.add_argument('--timestep-ratio', type=float, default=1.0, help='Final-step size ratio used in training.')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--temperature', type=float, default=1.0, help='Positive mixture-logit temperature.')
    parser.add_argument('--cpu-offload', action='store_true', help='Offload model components to CPU to reduce VRAM.')
    parser.add_argument('--local-files-only', action='store_true', help='Do not download model files.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nfe < 2:
        raise ValueError('IBFlow-Qwen is trained for at least 2 NFEs.')
    if args.timestep_ratio <= 0:
        raise ValueError('--timestep-ratio must be positive.')
    if args.height <= 0 or args.width <= 0:
        raise ValueError('--height and --width must be positive.')
    if args.height % 16 or args.width % 16:
        raise ValueError('--height and --width must be divisible by 16 for Qwen-Image.')
    if args.temperature <= 0:
        raise ValueError('--temperature must be positive.')

    adapter_path = Path(args.adapter)
    if adapter_path.is_file() and adapter_path.suffix == '.pth':
        raise ValueError(
            '--adapter expects an exported adapter directory, not a training '
            '.pth checkpoint. Run export_ibflow_to_diffusers.py first.')

    dtype = getattr(torch, args.dtype)
    pipe = IBQwenImagePipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        revision=args.base_revision,
        local_files_only=args.local_files_only)

    adapter_kwargs = dict(
        target_module_name='transformer',
        revision=args.adapter_revision,
        use_safetensors=True,
        local_files_only=args.local_files_only)
    if args.adapter_subfolder:
        adapter_kwargs['subfolder'] = args.adapter_subfolder
    pipe.load_ibflow_adapter(args.adapter, **adapter_kwargs)

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=3.2,
        shift_terminal=None,
        use_dynamic_shifting=False)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        num_images_per_prompt=1,
        width=args.width,
        height=args.height,
        num_inference_steps=args.nfe,
        generator=generator,
        timestep_ratio=args.timestep_ratio,
        temperature=args.temperature,
    ).images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f'Saved image to {output_path}')


if __name__ == '__main__':
    main()
