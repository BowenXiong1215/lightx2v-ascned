#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-$(pwd)}"
TARGET_ROOT="$(cd "${TARGET_ROOT}" && pwd)"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

failed=0
while IFS=$'\t' read -r _ expected relative; do
  target="${TARGET_ROOT}/${relative}"
  if ! test -f "${target}" || test "$(sha256_file "${target}")" != "${expected}"; then
    echo "FAILED: ${relative}" >&2
    failed=1
  fi
done < "${PATCH_ROOT}/modified.tsv"

while IFS=$'\t' read -r expected relative; do
  target="${TARGET_ROOT}/${relative}"
  if ! test -f "${target}" || test "$(sha256_file "${target}")" != "${expected}"; then
    echo "FAILED: ${relative}" >&2
    failed=1
  fi
done < "${PATCH_ROOT}/added.tsv"
test "${failed}" -eq 0

python -m py_compile \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/runtime/accelerator.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/runtime/distributed.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/runtime/fsdp.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/runtime/ddp.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/model_zoo/base.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/model_zoo/minimax_h3/minimax_h3_t2av.py" \
  "${TARGET_ROOT}/lightx2v_train/lightx2v_train/trainers/dmd/runtime.py" \
  "${TARGET_ROOT}/lightx2v_train/train.py"

grep -q 'backend: hccl' "${TARGET_ROOT}/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml"
grep -q 'num_inference_steps: 4' "${TARGET_ROOT}/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml"
grep -q 'attention_backend: _native_npu' "${TARGET_ROOT}/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml"
grep -q 'student_init_lora_path' "${TARGET_ROOT}/lightx2v_train/lightx2v_train/trainers/dmd/runtime.py"

echo "Static verification: PASS"
