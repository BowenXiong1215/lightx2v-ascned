"""Memory-bounded gradient operations for accelerator training."""

import torch
import torch.distributed as dist

from lightx2v_train.runtime.accelerator import empty_cache, get_accelerator_type
from lightx2v_train.runtime.distributed import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)


def _local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _is_sharded(tensor):
    placements = getattr(tensor, "placements", ())
    return any(getattr(placement, "is_shard", lambda: False)() for placement in placements)


@torch.no_grad()
def chunked_clip_grad_norm_(parameters, max_norm, chunk_numel=262144):
    """Clip a distributed L2 gradient norm with bounded temporary memory.

    FSDP2 gradients are DTensors. Their local shards contribute to a scalar
    sum of squares which is reduced over the data-parallel mesh. Replicated
    gradients are divided by the mesh size before that reduction so they are
    counted once. Float32 conversion is limited to ``chunk_numel`` values.
    """

    parameters = list(parameters)
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    if float(max_norm) < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}.")
    if int(chunk_numel) <= 0:
        raise ValueError(f"chunk_numel must be positive, got {chunk_numel}.")

    empty_cache()
    first = _local_tensor(grads[0])
    total_sq = torch.zeros((), device=first.device, dtype=torch.float32)
    dp_size = get_data_parallel_world_size() if dist.is_initialized() else 1
    for grad in grads:
        local = _local_tensor(grad.detach()).reshape(-1)
        grad_sq = torch.zeros_like(total_sq)
        for chunk in local.split(int(chunk_numel)):
            work = torch.empty_like(chunk, dtype=torch.float32)
            work.copy_(chunk)
            work.square_()
            grad_sq.add_(work.sum())
            del work
        if dp_size > 1 and not _is_sharded(grad):
            grad_sq.div_(dp_size)
        total_sq.add_(grad_sq)

    if dp_size > 1:
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
    total_norm = total_sq.sqrt()
    clip_coefficient = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        _local_tensor(grad).mul_(clip_coefficient.to(dtype=_local_tensor(grad).dtype))
    return total_norm


def clip_grad_norm_(parameters, max_norm):
    if get_accelerator_type() == "npu":
        return chunked_clip_grad_norm_(parameters, max_norm)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
