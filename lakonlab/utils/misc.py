# Copyright (c) 2025 Hansheng Chen

import contextlib
import functools
import gc

import torch
from mmcv.parallel import is_module_wrapper
from peft.tuners.lora import LoraLayer


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if is_module_wrapper(obj):
            obj = obj.module
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if is_module_wrapper(pre):
        pre = pre.module
    return setattr(pre, post, val)


def rhasattr(obj, attr):
    return rgetattr(obj, attr, None) is not None


def rdelattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if is_module_wrapper(pre):
        pre = pre.module
    return delattr(pre, post)


class module_requires_grad:
    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        for p in self.module.parameters():
            self.prev.append(p.requires_grad)
            p.requires_grad = self.requires_grad

    def __exit__(self, exc_type, exc_value, traceback):
        for p, r in zip(self.module.parameters(), self.prev):
            p.requires_grad = r


class module_eval:
    def __init__(self, module):
        self.module = module
        self.prev = None

    def __enter__(self):
        self.prev = self.module.training
        self.module.train(False)

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.train(self.prev)


def all_frozen(modules):
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                return False
    return True


def tie_untrained_submodules(tgt_module, src_module, tie_tgt_lora_base_layer=False):
    for key, src_submodule in src_module._modules.items():
        if key in tgt_module._modules:
            if (tie_tgt_lora_base_layer
                    and isinstance(tgt_module._modules[key], LoraLayer)
                    and not isinstance(src_submodule, LoraLayer)):
                if all_frozen((tgt_module._modules[key]._modules['base_layer'], src_submodule)):
                    tgt_module._modules[key]._modules['base_layer'] = src_submodule
                else:
                    tie_untrained_submodules(
                        tgt_module._modules[key]._modules['base_layer'], src_submodule, tie_tgt_lora_base_layer)
            else:
                if all_frozen((tgt_module._modules[key], src_submodule)):
                    tgt_module._modules[key] = src_submodule
                else:
                    tie_untrained_submodules(
                        tgt_module._modules[key], src_submodule, tie_tgt_lora_base_layer)


def clone_params(tgt_module, src_module, recursive=True):
    """Clone parameters and buffers from src_module to tgt_module (sharing the same structure).
    Tied parameters/buffers are not cloned. Used for EMA model initialization.
    """
    for key, val in src_module._parameters.items():
        if (val is not None) \
                and (val is not tgt_module._parameters[key]):
            tgt_module._parameters[key] = val.clone()
    for key, val in src_module._buffers.items():
        if val is not tgt_module._buffers[key]:
            tgt_module._buffers[key] = val.clone()
    if recursive:
        for key, val in src_module._modules.items():
            clone_params(
                tgt_module._modules[key], val, recursive)


@contextlib.contextmanager
def gc_context(enable=False):
    prev_enabled = gc.isenabled()
    if enable:
        gc.enable()
    else:
        gc.disable()
    try:
        yield
    finally:
        if prev_enabled:
            gc.enable()
        else:
            gc.disable()
