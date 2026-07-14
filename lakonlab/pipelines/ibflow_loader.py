# Copyright (c) 2025 Hansheng Chen

import json
import os
from typing import Union, Optional

import torch
import accelerate
import diffusers
from safetensors import safe_open
from diffusers.models import AutoModel
from diffusers.models.modeling_utils import (
    load_state_dict,
    _LOW_CPU_MEM_USAGE_DEFAULT,
    no_init_weights,
    ContextManagers
)
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from lakonlab.models.architecture.ibflow.ibqwen import _IBQwenImageTransformer2DModel


_SET_ADAPTER_SCALE_FN_MAPPING.update(
    _IBQwenImageTransformer2DModel=lambda model_cls, weights: weights,
)

logger = logging.get_logger(__name__)


class IBFlowLoaderMixin:

    def load_ibflow_adapter(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        target_module_name: str = "transformer",
        adapter_name: Optional[str] = None,
        **kwargs
    ):
        r"""
        Load an IBFlow adapter from a pretrained model repository into the target module.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            target_module_name (`str`, *optional*, defaults to `"transformer"`):
                The module name in the model to load the IBFlow adapter into.
            adapter_name (`str`, *optional*):
                The name to assign to the loaded adapter. If not provided, it defaults to
                `"{target_module_name}_ibflow"`.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.
            disable_mmap ('bool', *optional*, defaults to 'False'):
                Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

        Returns:
            `str` or `None`: The name assigned to the loaded adapter, or `None` if no LoRA weights were found.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        disable_mmap = kwargs.pop("disable_mmap", False)

        if use_safetensors is None:
            use_safetensors = True
        if not use_safetensors:
            raise ValueError('IBFlow adapters must use safetensors serialization.')
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected adapter loader arguments: {unknown}.')

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        user_agent = {
            "diffusers": diffusers.__version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # 1. Read architecture parameters from the adapter config. This
        # Qwen-only release always instantiates the local IBFlow class, which
        # also keeps previously exported adapters compatible after renaming.
        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        config = AutoModel.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **load_config_kwargs)
        source_class_name = config.get('_class_name')
        allowed_class_names = {
            'ArcQwenImageTransformer2DModel',
            'IBQwenImageTransformer2DModel',
            '_ArcQwenImageTransformer2DModel',
            '_IBQwenImageTransformer2DModel',
        }
        if source_class_name not in allowed_class_names:
            raise ValueError(
                f'Unsupported adapter class {source_class_name!r}; expected one of '
                f'{sorted(allowed_class_names)}.')
        model_cls = _IBQwenImageTransformer2DModel

        # 2. Get model file

        model_file = None

        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )

            except IOError as e:
                logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                raise

        # 3. Initialize model

        base_module = getattr(self, target_module_name)

        torch_dtype = base_module.dtype
        device = base_module.device
        dtype_orig = model_cls._set_default_torch_dtype(torch_dtype)

        init_contexts = [no_init_weights()]

        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights())

        with ContextManagers(init_contexts):
            ibflow_module = model_cls.from_config(config).eval()

        torch.set_default_dtype(dtype_orig)

        # 4. Load model weights

        if model_file is not None:
            base_state_dict = base_module.state_dict()
            lora_state_dict = dict()

            adapter_state_dict = load_state_dict(model_file, disable_mmap=disable_mmap)
            required_output_keys = {
                'norm_out.linear.bias',
                'norm_out.linear.weight',
                'proj_out_loggamma.bias',
                'proj_out_loggamma.weight',
                'proj_out_logweights.bias',
                'proj_out_logweights.weight',
                'proj_out_means.bias',
                'proj_out_means.weight',
            }
            prefix = f'{target_module_name}.'
            normalized_keys = {
                key.removeprefix(prefix) for key in adapter_state_dict
            }
            missing_output_keys = required_output_keys - normalized_keys
            if missing_output_keys:
                raise ValueError(
                    'Adapter is missing required IBFlow output tensors: '
                    f'{sorted(missing_output_keys)}.')

            lora_keys = {key for key in normalized_keys if 'lora_' in key}
            lora_a_stems = {
                key.removesuffix('.lora_A.weight')
                for key in lora_keys if key.endswith('.lora_A.weight')
            }
            lora_b_stems = {
                key.removesuffix('.lora_B.weight')
                for key in lora_keys if key.endswith('.lora_B.weight')
            }
            if not lora_a_stems or lora_a_stems != lora_b_stems:
                raise ValueError(
                    'Adapter LoRA A/B tensors are missing or unpaired. '
                    f'A-only={sorted(lora_a_stems - lora_b_stems)}, '
                    f'B-only={sorted(lora_b_stems - lora_a_stems)}.')

            module_state_dict = ibflow_module.state_dict()
            for key, value in adapter_state_dict.items():
                normalized_key = key.removeprefix(prefix)
                if 'lora_' in normalized_key:
                    continue
                if normalized_key not in module_state_dict:
                    raise ValueError(f'Unexpected non-LoRA adapter tensor: {normalized_key}.')
                if module_state_dict[normalized_key].shape != value.shape:
                    raise ValueError(
                        f'Shape mismatch for {normalized_key}: adapter '
                        f'{tuple(value.shape)} vs model '
                        f'{tuple(module_state_dict[normalized_key].shape)}.')

            adapter_metadata = {}
            if str(model_file).endswith('.safetensors'):
                with safe_open(model_file, framework='pt', device='cpu') as handle:
                    adapter_metadata = handle.metadata() or {}
            policy_config_raw = adapter_metadata.get('policy_config')
            if policy_config_raw:
                try:
                    policy_config = json.loads(policy_config_raw)
                except json.JSONDecodeError as exc:
                    raise ValueError('Invalid policy_config JSON in adapter metadata.') from exc
                if policy_config.get('type') not in {'ArcFlow', 'IBFlow'}:
                    raise ValueError(
                        f'Unsupported adapter policy type: {policy_config.get("type")!r}.')
            self._ibflow_adapter_metadata = {
                **adapter_metadata,
                'source_class_name': source_class_name,
            }

            for k in adapter_state_dict.keys():
                adapter_state_dict[k] = adapter_state_dict[k].to(dtype=torch_dtype, device=device)
                if "lora" in k:
                    lora_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]
                else:
                    base_state_dict[k.removeprefix(f"{target_module_name}.")] = adapter_state_dict[k]

            if len(lora_state_dict) == 0:
                raise ValueError(
                    f'No LoRA weights were found in {pretrained_model_name_or_path}.')

            else:
                if adapter_name is None:
                    adapter_name = f"{target_module_name}_ibflow"

                ibflow_module.load_state_dict(
                    base_state_dict, strict=False, assign=True)
                ibflow_module.load_lora_adapter(
                    lora_state_dict, prefix=None, adapter_name=adapter_name)

                setattr(self, target_module_name, ibflow_module)

        else:
            adapter_name = None

        return adapter_name
    
