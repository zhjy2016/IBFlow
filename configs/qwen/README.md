## Distilling Qwen-Image

### Data Preparation

Precompute prompt embeddings from
[Lakonik/t2i-prompts-3m](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m) on one node with 8
GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  cache_image_prompt_data.py configs/qwen/ibflow_qwen_2nfe_k16.py \
  --text-encoder configs/qwen/_text_encoder.py \
  --max-size 2304128 \
  --launcher pytorch \
  --diff_seed
```

The cache is saved to `data/t2i_prompts_3m/preproc_qwen/` by default and requires approximately
380GB of storage. Change `data_root` in `_data_trainval.py` to use another location.

### Training

Run the IBFlow configuration on one node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 \
  train.py configs/qwen/ibflow_qwen_2nfe_k16.py \
  --launcher pytorch \
  --diff_seed
```

The default configuration uses four samples per GPU, accumulates gradients over four student
steps, and requires approximately 70GB of memory per GPU. Eight A100-80GB GPUs are recommended.
