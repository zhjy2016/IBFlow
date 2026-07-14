# IB-Flow

<a href="https://arxiv.org/pdf/2607.09133v1"><img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper: arXiv"></a> <a href="https://huggingface.co/ChangyuanWang/IBFlow-Qwen-2NFE"><img src="https://img.shields.io/badge/HuggingFace-Model-orange" alt="Hugging Face model"></a> <a href="https://github.com/zhjy2016/IBFlow"><img src="https://img.shields.io/badge/GitHub-Code-blue" alt="GitHub code"></a>

**IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation**

Yiting Wang<sup>1</sup>,
Jingyi Zhang<sup>2</sup>,
Wenhu Zhang<sup>3</sup>,
Ke Chao<sup>4</sup>,
Yves Liang<sup>1</sup>,
Kun Cheng<sup>2</sup>,
Kang Zhao<sup>2</sup>

<sup>1</sup>Tsinghua University;
<sup>2</sup>Wan Team, Alibaba Group;
<sup>3</sup>HKUST;
<sup>4</sup>Beijing Normal University

## Overview

IB-Flow enables high-quality text-to-image generation with only <b>2 NFEs</b>.

Few-step distillation greatly reduces the inference cost of diffusion models, but fixed guidance
does not account for how generation changes from global structure formation to detail refinement.
IB-Flow formulates CFG distillation from an information-bottleneck perspective, combining
instance-aware supervision targets with an entropy-aware guidance schedule. This repository
provides the Qwen-Image inference, prompt preprocessing, and distributed training code.

## News

- `[2026-07-14]` The pretrained [IBFlow-Qwen-2NFE](https://huggingface.co/ChangyuanWang/IBFlow-Qwen-2NFE) adapter is released.
- `[2026-07-13]` The paper is available on [arXiv](https://arxiv.org/pdf/2607.09133v1).
- `[2026-07-12]` 🔥 The Qwen-Image training and inference code for IB-Flow is released.

## Quickstart

### Environment Setup

```bash
conda create -y -n ibflow python=3.10 ninja
conda activate ibflow

pip install torch==2.6.0 torchvision==0.21.0

pip install -r requirements.txt --no-build-isolation
```

### Inference

IBFlow-Qwen uses [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) as its base model. The
released adapter is designed for 2-NFE inference with `timestep_ratio=1.0`; both are the defaults
used by the command below. `IBFLOW_MODEL` may be a Hugging Face model ID or a local adapter
directory.

```bash
export QWEN_IMAGE_MODEL=Qwen/Qwen-Image
export IBFLOW_MODEL=ChangyuanWang/IBFlow-Qwen-2NFE

python inference_qwen.py \
  --base-model "${QWEN_IMAGE_MODEL}" \
  --adapter "${IBFLOW_MODEL}" \
  --prompt "A cinematic close-up portrait of an elderly Tibetan artisan in a sunlit workshop, weathered skin, fine wrinkles, warm brown eyes, both hands carefully carving a small wooden bird, realistic anatomy, shallow depth of field, 85mm photography." \
  --output outputs/ibflow_qwen.png
```

### Data Preparation

Training uses prompts from
[Lakonik/t2i-prompts-3m](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m). Since loading the
text encoder together with the teacher and student requires additional GPU memory, we precompute
the prompt embeddings on one node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  cache_image_prompt_data.py configs/qwen/ibflow_qwen_2nfe_k16.py \
  --text-encoder configs/qwen/_text_encoder.py \
  --max-size 2304128 \
  --launcher pytorch \
  --diff_seed
```

The cache is saved to `data/t2i_prompts_3m/preproc_qwen/` by default and requires approximately
380GB of storage. Change `data_root` in `configs/qwen/_data_trainval.py` to use another location.

### Training

Run the IB-Flow configuration on one node with 8 GPUs. The default configuration uses four samples
per GPU and requires approximately 70GB of memory per GPU, so 8 A100-80GB GPUs are recommended.

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  train.py configs/qwen/ibflow_qwen_2nfe_k16.py \
  --launcher pytorch \
  --diff_seed
```

Checkpoints are saved to `checkpoints/<experiment_name>/iter_*.pth`, while training logs and
TensorBoard events are written to `work_dirs/<experiment_name>/`. The experiment name is defined by
`name` in the training config.

Visualize the training losses with TensorBoard:

```bash
tensorboard --logdir work_dirs/
```

### Exporting a Trained Model

Training checkpoints are `.pth` files. Export the EMA adapter to the safetensors format used by the
inference pipeline:

```bash
export IBFLOW_CHECKPOINT=/path/to/iter_5000.pth

python export_ibflow_to_diffusers.py \
  configs/qwen/ibflow_qwen_2nfe_k16.py \
  --ckpt "${IBFLOW_CHECKPOINT}" \
  --out-dir outputs/ibflow_qwen_adapter
```

Use the exported adapter for inference:

```bash
python inference_qwen.py \
  --base-model Qwen/Qwen-Image \
  --adapter outputs/ibflow_qwen_adapter \
  --prompt "A cinematic close-up portrait of an elderly Tibetan artisan in a sunlit workshop, weathered skin, fine wrinkles, warm brown eyes, both hands carefully carving a small wooden bird, realistic anatomy, shallow depth of field, 85mm photography." \
  --output outputs/ibflow_qwen_custom.png
```

## Acknowledgments

This project benefits from the open-source work of
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image),
[ArcFlow](https://github.com/pnotp/ArcFlow),
[pi-Flow](https://github.com/Lakonik/piFlow), and
[TwinFlow](https://github.com/inclusionAI/TwinFlow). The training prompts are provided by
[Lakonik/t2i-prompts-3m](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m). We thank their
authors and maintainers for making these resources publicly available.

## Citation

If you find IB-Flow useful in your research, please cite:

```bibtex
@misc{wang2026ibflow,
  title={IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation},
  author={Wang, Yiting and Zhang, Jingyi and Zhang, Wenhu and Chao, Ke and Liang, Yves and Cheng, Kun and Zhao, Kang},
  year={2026},
  eprint={2607.09133},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
