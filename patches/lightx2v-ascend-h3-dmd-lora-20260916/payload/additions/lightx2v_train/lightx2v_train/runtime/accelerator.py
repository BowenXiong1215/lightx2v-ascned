"""Small accelerator abstraction used by the training runtime.

The inference package already has a platform layer, but importing it also
initializes a device at import time.  Training needs configuration-driven
initialization before the process group and device mesh are constructed, so
the helpers here deliberately have no import-time side effects.
"""

import importlib
import os

import torch


_ACCELERATOR_TYPE = None
_SUPPORTED_ACCELERATORS = {"cpu", "cuda", "npu"}


def resolve_accelerator_type(config=None):
    distributed = (config or {}).get("distributed", {})
    configured = distributed.get("device_type") or distributed.get("accelerator")
    if configured is None:
        platform = os.getenv("PLATFORM", "").lower()
        if distributed.get("backend") == "hccl" or platform in {"ascend", "ascend_npu", "npu"}:
            configured = "npu"
        elif torch.cuda.is_available():
            configured = "cuda"
        else:
            configured = "cpu"
    configured = str(configured).lower()
    if configured == "ascend_npu":
        configured = "npu"
    if configured not in _SUPPORTED_ACCELERATORS:
        raise ValueError(
            f"Unsupported distributed.device_type={configured!r}; "
            f"expected one of {sorted(_SUPPORTED_ACCELERATORS)}."
        )
    return configured


def _load_npu_runtime():
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError(
            "Ascend training requires torch_npu. Install the torch/torch_npu "
            "pair matching the CANN image before launching with backend=hccl."
        ) from error
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu was imported but torch.npu is unavailable.")


def initialize_accelerator(config=None, local_rank=0):
    global _ACCELERATOR_TYPE
    accelerator_type = resolve_accelerator_type(config)
    if accelerator_type == "npu":
        _load_npu_runtime()
    accelerator = getattr(torch, accelerator_type, None)
    if accelerator_type != "cpu":
        if accelerator is None or not accelerator.is_available():
            raise RuntimeError(f"Requested {accelerator_type} accelerator is not available.")
        accelerator.set_device(int(local_rank))
    _ACCELERATOR_TYPE = accelerator_type
    return accelerator_type


def get_accelerator_type():
    return _ACCELERATOR_TYPE or resolve_accelerator_type()


def get_accelerator():
    accelerator_type = get_accelerator_type()
    return None if accelerator_type == "cpu" else getattr(torch, accelerator_type)


def get_device():
    accelerator_type = get_accelerator_type()
    if accelerator_type == "cpu":
        return torch.device("cpu")
    accelerator = get_accelerator()
    return torch.device(accelerator_type, accelerator.current_device())


def empty_cache():
    accelerator = get_accelerator()
    if accelerator is not None and hasattr(accelerator, "empty_cache"):
        accelerator.empty_cache()


def manual_seed_all(seed):
    accelerator = get_accelerator()
    if accelerator is not None and hasattr(accelerator, "manual_seed_all"):
        accelerator.manual_seed_all(seed)


def memory_gb():
    accelerator = get_accelerator()
    if accelerator is None:
        return None
    return accelerator.memory_allocated() / 1024**3, accelerator.memory_reserved() / 1024**3

