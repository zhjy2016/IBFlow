# Copyright (c) 2025 Hansheng Chen

import os
import time
from functools import wraps
from io import BytesIO

import imageio
import numpy as np
import torch.distributed as dist
from huggingface_hub import hf_hub_download


def retry(tries=5, delay=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == tries:
                        raise
                    print(f"Attempt {attempt} failed: {exc}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


@retry()
def download_from_huggingface(filename):
    parts = filename.removeprefix('huggingface://').split('/')
    repo_id = '/'.join(parts[:2])
    repo_filename = '/'.join(parts[2:])
    is_dist = dist.is_available() and dist.is_initialized()
    local_rank = dist.get_node_local_rank() if is_dist else 0
    if local_rank == 0:
        cached_file = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    if is_dist:
        dist.barrier()
    if local_rank > 0:
        cached_file = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    return cached_file


def load_image(filepath, file_client):
    img_bytes = file_client.get(filepath)
    extension = os.path.splitext(filepath)[-1].lower()
    arr = imageio.v3.imread(BytesIO(img_bytes), extension=extension)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr
