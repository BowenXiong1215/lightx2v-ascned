#!/usr/bin/env bash

set -euo pipefail

H3_CONFIG_PATH="${H3_CONFIG_PATH:-configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml}"
H3_NUM_PROCESSES="${H3_NUM_PROCESSES:-8}"

export PLATFORM="${PLATFORM:-ascend_npu}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"

python - <<'PY'
import torch
import torch_npu

if not torch.npu.is_available():
    raise SystemExit("torch_npu imported, but no Ascend NPU is available")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"npu_count={torch.npu.device_count()}")
PY

torchrun \
    --standalone \
    --nproc_per_node="${H3_NUM_PROCESSES}" \
    train.py \
    --config "${H3_CONFIG_PATH}"
