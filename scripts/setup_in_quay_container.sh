#!/usr/bin/env bash
set -euo pipefail

PATCH_REPO=/workspace/lightx2v-ascned
LIGHTX2V_ROOT=/workspace/LightX2V
LIGHTX2V_SHA=bb8301a7dfe8180772bb864d269d8bd05a938f8e

test -f "${PATCH_REPO}/UPSTREAM_COMMIT" || {
  echo "Mount the lightx2v-ascned repository at ${PATCH_REPO}" >&2
  exit 2
}
test "$(cat "${PATCH_REPO}/UPSTREAM_COMMIT")" = "${LIGHTX2V_SHA}" || {
  echo "Patch repository targets a different LightX2V commit" >&2
  exit 2
}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates git ffmpeg
rm -rf /var/lib/apt/lists/*

# Freeze the Ascend base image's PyTorch packages before pip resolves the
# remaining training dependencies.
python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path

constraints = []
for name in ("torch", "torch-npu", "torchvision", "torchaudio"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        if name in ("torch", "torch-npu"):
            raise RuntimeError(f"The Quay base image is missing {name}")
    else:
        constraints.append(f"{name}=={version}")
Path("/tmp/ascend-torch-constraints.txt").write_text("\n".join(constraints) + "\n")
print("Keeping base image framework packages:", constraints)
PY

python -m pip install --no-cache-dir -c /tmp/ascend-torch-constraints.txt \
  'diffusers==0.40.0' 'transformers==4.57.6' 'peft==0.18.0' \
  accelerate omegaconf safetensors loguru einops imageio pillow \
  ftfy sentencepiece protobuf tqdm requests scipy numpy packaging

if test ! -e "${LIGHTX2V_ROOT}"; then
  git clone https://github.com/ModelTC/LightX2V.git "${LIGHTX2V_ROOT}"
fi
test -d "${LIGHTX2V_ROOT}/.git" || {
  echo "${LIGHTX2V_ROOT} exists but is not a Git checkout" >&2
  exit 2
}
git -C "${LIGHTX2V_ROOT}" checkout --detach "${LIGHTX2V_SHA}"
test "$(git -C "${LIGHTX2V_ROOT}" rev-parse HEAD)" = "${LIGHTX2V_SHA}"

PATCH_DIR="${PATCH_REPO}/patches/lightx2v-ascend-h3-dmd-lora-20260916"
bash "${PATCH_DIR}/install.sh" "${LIGHTX2V_ROOT}"
bash "${PATCH_DIR}/verify.sh" "${LIGHTX2V_ROOT}"
python -m pip install --no-cache-dir --no-deps -e "${LIGHTX2V_ROOT}"

python - <<'PY'
import torch
import torch_npu
import diffusers
from diffusers import MiniMaxH3Transformer3DModel

assert torch.__version__.startswith("2.7.1"), torch.__version__
assert torch_npu.__version__.startswith("2.7.1"), torch_npu.__version__
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("diffusers:", diffusers.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count() if torch.npu.is_available() else 0)
PY

echo "Setup complete. Run H3 DMD-LoRA from ${LIGHTX2V_ROOT}/lightx2v_train"
