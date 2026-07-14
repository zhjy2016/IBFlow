# Copyright (c) 2025 Hansheng Chen

import os
import os.path as osp
import logging
import time
import re
import torch
import torch.nn as nn
import torch.distributed as dist
import mmcv
from typing import Union, Callable, Optional
from collections import OrderedDict
from torch.optim import Optimizer
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_optimizer_state_dict)
from safetensors.torch import load_file, load
from diffusers.utils.hub_utils import _get_checkpoint_shard_files
from mmcv.runner import CheckpointLoader, get_dist_info, _load_checkpoint
from mmcv.parallel import is_module_wrapper
from lakonlab.utils import download_from_huggingface, rgetattr


def load_full_state_dict(module: nn.Module,
                         state_dict: Union[dict, OrderedDict],
                         strict: bool = False,
                         logger: Optional[logging.Logger] = None,
                         assign: bool = False) -> None:
    unexpected_keys: List[str] = []
    all_missing_keys: List[str] = []
    err_msg: List[str] = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()  # type: ignore
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # Recursively unwrap parallel modules before loading.
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata['assign_to_params_buffers'] = assign
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore

    # ignore "num_batches_tracked" of BN layers
    missing_keys = []
    for key in all_missing_keys:
        if not rgetattr(module, key + '._fsdp_flattened', False):
            missing_keys.append(key)

    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def exists_ckpt(filename):
    if not filename:
        return False
    return os.path.exists(filename)


@CheckpointLoader.register_scheme(prefixes='huggingface://')
def load_from_huggingface(filename, map_location=None):
    cached_file = download_from_huggingface(filename)
    if cached_file.endswith('.index.json'):  # sharded checkpoint
        filename = filename.replace('huggingface://', '').split('/')
        repo_id = '/'.join(filename[:2])
        repo_subfolder = '/'.join(filename[2:-1])
        is_dist = dist.is_available() and dist.is_initialized()
        if is_dist:
            local_rank = dist.get_node_local_rank()
        else:
            local_rank = 0
        if local_rank == 0:
            sharded_cached_files = _get_checkpoint_shard_files(
                repo_id,
                cached_file,
                subfolder=repo_subfolder)[0]
        if is_dist:
            dist.barrier()
        if local_rank > 0:
            sharded_cached_files = _get_checkpoint_shard_files(
                repo_id,
                cached_file,
                subfolder=repo_subfolder)[0]
        ckpt = OrderedDict()
        for sharded_cached_file in sharded_cached_files:
            ext = os.path.splitext(sharded_cached_file)[-1].lower()
            if ext == '.safetensors':
                ckpt.update(load_file(sharded_cached_file, device=map_location))
            else:
                ckpt.update(torch.load(sharded_cached_file, map_location=map_location))
        return ckpt
    else:
        ext = os.path.splitext(cached_file)[-1].lower()
        if ext == '.safetensors':
            return load_file(cached_file, device=map_location)
        else:
            return torch.load(cached_file, map_location=map_location)


@CheckpointLoader.register_scheme(prefixes='', force=True)
def load_from_local(filename, map_location=None):
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')

    # ================= 新增：支持读取本地的 .index.json 分片权重 =================
    if filename.endswith('.index.json'):
        import json
        with open(filename, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        # 获取所有去重后的分片文件名
        shards = set(weight_map.values())
        base_dir = osp.dirname(filename)
        
        ckpt = OrderedDict()
        for shard_file in shards: # 遍历加载所有的 safetensors 分片
            shard_path = osp.join(base_dir, shard_file)
            ext = osp.splitext(shard_path)[-1].lower()
            if ext == '.safetensors':
                shard_ckpt = load_file(shard_path, device=map_location)
            else:
                shard_ckpt = torch.load(shard_path, map_location=map_location)
            # 在内存中合并所有的权重字典
            ckpt.update(shard_ckpt)
        return ckpt
        
    ext = os.path.splitext(filename)[-1].lower()
    if ext == '.safetensors':
        with open(filename, "rb") as f:  # load_file may fail with FUSE/NFS mmap
            ckpt = load(f.read())
            if map_location is not None:
                for k in ckpt:
                    ckpt[k] = ckpt[k].to(map_location)
    else:
        ckpt = torch.load(filename, map_location=map_location)
    return ckpt


def load_checkpoint(model: torch.nn.Module,
                    filename: str,
                    map_location: Union[str, Callable, None] = None,
                    strict: bool = False,
                    logger: Optional[logging.Logger] = None,
                    revise_keys: list = [(r'^module\.', '')],
                    assign: bool = False) -> Union[dict, OrderedDict]:
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_full_state_dict(model, state_dict, strict, logger, assign)
    return checkpoint


def _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=False, cpu_offload=False):
    for name, param in module._parameters.items():
        if param is not None and (not trainable_only or param.requires_grad):
            if not keep_vars:
                param = param.detach()
            if cpu_offload:
                param = param.cpu()
            destination[prefix + name] = param
    for name, buf in module._buffers.items():
        if buf is not None:
            if not keep_vars:
                buf = buf.detach()
            if cpu_offload:
                buf = buf.cpu()
            destination[prefix + name] = buf


def get_state_dict(module,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   trainable_only=False,
                   cpu_offload=True):
    if is_module_wrapper(module):
        module = module.module
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    local_metadata = dict(version=module._version)
    if hasattr(destination, '_metadata'):
        destination._metadata[prefix[:-1]] = local_metadata
    for hook in module._state_dict_pre_hooks.values():
        hook(module, prefix, keep_vars)
    _save_to_state_dict(
        module, destination, prefix, keep_vars,
        trainable_only=trainable_only, cpu_offload=cpu_offload)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars,
                trainable_only=trainable_only, cpu_offload=cpu_offload)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def get_optim_state_dict(model, optimizer, bf16=False):
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True))
    if 'state' in optim_state_dict:
        for state_name, state in optim_state_dict['state'].items():
            new_state = dict()
            for k, v in state.items():
                if bf16 and isinstance(v, torch.Tensor) and v.dtype == torch.float32 and v.numel() > 1:
                    v = v.to(dtype=torch.bfloat16)
                new_state[k] = v
            optim_state_dict['state'][state_name] = new_state
    return optim_state_dict


def write_checkpoint_to_file(checkpoint, filepath, create_symlink=False, after_save_hook=None):
    mmcv.mkdir_or_exist(osp.dirname(filepath))
    with open(filepath, 'wb') as file:
        torch.save(checkpoint, file)
        file.flush()
    if create_symlink:
        dst_file = osp.join(osp.dirname(filepath), 'latest.pth')
        mmcv.symlink(osp.basename(filepath), dst_file)
    if after_save_hook is not None:
        after_save_hook()


def get_checkpoint(model,
                   optimizer=None,
                   loss_scaler=None,
                   meta=None,
                   trainable_only=False,
                   fp16=False,
                   fp16_ema=False,
                   bf16_optim=False):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': get_state_dict(model, trainable_only=trainable_only, cpu_offload=True)}
    if fp16 or fp16_ema:
        for k, v in checkpoint['state_dict'].items():
            if ((fp16 and '_ema.' not in k and '_ema2.' not in k) or (fp16_ema and ('_ema.' in k or '_ema2.' in k))) \
                    and v.dtype == torch.float32:
                checkpoint['state_dict'][k] = v.half()

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = get_optim_state_dict(model, optimizer, bf16_optim)
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            submodule = getattr(model, name)
            checkpoint['optimizer'][name] = get_optim_state_dict(submodule, optim, bf16_optim)

    # save loss scaler for mixed-precision (FP16) training
    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    return checkpoint
