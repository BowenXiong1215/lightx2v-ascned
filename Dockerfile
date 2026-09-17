# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11@sha256:b69d14f39d47ec8f66e635f16190788e3c2b2174d0994f63090afcdf885b06f1
FROM ${BASE_IMAGE}

ARG LIGHTX2V_SHA=bb8301a7dfe8180772bb864d269d8bd05a938f8e
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PLATFORM=ascend_npu \
    HCCL_CONNECT_TIMEOUT=7200 \
    HCCL_EXEC_TIMEOUT=7200

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Keep the torch/torch_npu pair supplied by the CANN image. Pip resolves all
# training dependencies against these exact installed versions.
RUN python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path

names = ("torch", "torch-npu", "torchvision", "torchaudio")
constraints = []
for name in names:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        if name in ("torch", "torch-npu"):
            raise RuntimeError(f"Base image is missing {name}")
    else:
        constraints.append(f"{name}=={version}")
Path("/tmp/ascend-torch-constraints.txt").write_text("\n".join(constraints) + "\n")
print("Base image framework constraints:", constraints)
PY

RUN python -m pip install --no-cache-dir -c /tmp/ascend-torch-constraints.txt \
      'diffusers==0.40.0' 'transformers==5.14.1' 'peft==0.21.0' \
      'accelerate==1.14.0' 'huggingface-hub>=1.5,<2' \
      omegaconf safetensors loguru einops imageio pillow \
      ftfy sentencepiece protobuf tqdm requests scipy numpy packaging

# Qwen3-VL's video processor imports torchvision. Keep the base image's
# torch/torch_npu pair intact while installing the matching ARM/AMD64 wheel.
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: 'torchvision==0.22.1'

WORKDIR /workspace
RUN git clone https://github.com/ModelTC/LightX2V.git LightX2V \
    && cd LightX2V \
    && git checkout --detach "${LIGHTX2V_SHA}" \
    && test "$(git rev-parse HEAD)" = "${LIGHTX2V_SHA}"

COPY patches/lightx2v-ascend-h3-dmd-lora-20260916 /workspace/patch
RUN bash /workspace/patch/install.sh /workspace/LightX2V \
    && bash /workspace/patch/verify.sh /workspace/LightX2V \
    && python -m pip install --no-cache-dir --no-deps -e /workspace/LightX2V \
    && python - <<'PY'
import torch
import torch_npu
import diffusers
import torchvision
from transformers import Qwen3VLVideoProcessor
from diffusers import MiniMaxH3Transformer3DModel

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch_npu.__version__.startswith("2.7.1"), torch_npu.__version__
print("H3 Ascend build imports: PASS", torch.__version__, torch_npu.__version__, diffusers.__version__)
PY

WORKDIR /workspace/LightX2V/lightx2v_train
CMD ["bash"]
