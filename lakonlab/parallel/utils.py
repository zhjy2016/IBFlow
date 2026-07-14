import mmcv
import torch

from . import MMDistributedDataParallel, DistributedDataParallelWrapper


def apply_module_wrapper(model, module_wrapper, cfg):
    wrapper = module_wrapper.lower()
    if wrapper == 'ddp':
        mmcv.print_log('Use DDP Wrapper.', 'mmgen')
        model = DistributedDataParallelWrapper(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False, find_unused_parameters=False)
    elif wrapper == 'mmddp':
        mmcv.print_log('Use MMDistributedDataParallel.', 'mmgen')
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False, find_unused_parameters=False)
    else:
        raise ValueError(f'Unsupported module wrapper: {module_wrapper}. Use ddp or mmddp.')
    return model
