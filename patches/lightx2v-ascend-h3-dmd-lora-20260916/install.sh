#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-$(pwd)}"
TARGET_ROOT="$(cd "${TARGET_ROOT}" && pwd)"
UPSTREAM_COMMIT=bb8301a7dfe8180772bb864d269d8bd05a938f8e

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

require_source_tree() {
  test -f "${TARGET_ROOT}/pyproject.toml" || {
    echo "Not a LightX2V source tree: ${TARGET_ROOT}" >&2
    exit 2
  }
  grep -q 'name = "lightx2v"' "${TARGET_ROOT}/pyproject.toml" || {
    echo "Unexpected pyproject.toml in ${TARGET_ROOT}" >&2
    exit 2
  }
}

validate_bundle() {
  local expected relative source actual
  while IFS=$'\t' read -r _ expected relative; do
    source="${PATCH_ROOT}/payload/replacements/${relative}"
    test -f "${source}" || { echo "Missing replacement payload: ${relative}" >&2; exit 3; }
    actual="$(sha256_file "${source}")"
    test "${actual}" = "${expected}" || { echo "Corrupt replacement payload: ${relative}" >&2; exit 3; }
  done < "${PATCH_ROOT}/modified.tsv"

  while IFS=$'\t' read -r expected relative; do
    source="${PATCH_ROOT}/payload/additions/${relative}"
    test -f "${source}" || { echo "Missing addition payload: ${relative}" >&2; exit 3; }
    actual="$(sha256_file "${source}")"
    test "${actual}" = "${expected}" || { echo "Corrupt addition payload: ${relative}" >&2; exit 3; }
  done < "${PATCH_ROOT}/added.tsv"
}

validate_target() {
  local upstream expected relative target actual accepted
  while IFS=$'\t' read -r upstream expected relative; do
    target="${TARGET_ROOT}/${relative}"
    test -f "${target}" || { echo "Missing upstream file: ${relative}" >&2; exit 4; }
    actual="$(sha256_file "${target}")"
    accepted=0
    if test -f "${PATCH_ROOT}/accepted_previous.tsv" && \
       awk -F '\t' -v hash="${actual}" -v path="${relative}" \
         '$1 == hash && $2 == path { found=1 } END { exit !found }' \
         "${PATCH_ROOT}/accepted_previous.tsv"; then
      accepted=1
    fi
    if test "${actual}" != "${upstream}" && test "${actual}" != "${expected}" && test "${accepted}" -ne 1; then
      echo "Source version mismatch: ${relative}" >&2
      echo "Expected upstream LightX2V commit ${UPSTREAM_COMMIT}" >&2
      exit 4
    fi
  done < "${PATCH_ROOT}/modified.tsv"

  while IFS=$'\t' read -r expected relative; do
    target="${TARGET_ROOT}/${relative}"
    if test -e "${target}" && test "$(sha256_file "${target}")" != "${expected}"; then
      echo "Existing addition differs: ${relative}" >&2
      exit 4
    fi
  done < "${PATCH_ROOT}/added.tsv"
}

apply_payload() {
  local relative source target
  while IFS=$'\t' read -r _ _ relative; do
    source="${PATCH_ROOT}/payload/replacements/${relative}"
    target="${TARGET_ROOT}/${relative}"
    mkdir -p "$(dirname "${target}")"
    cp -p "${source}" "${target}"
    echo "replaced: ${relative}"
  done < "${PATCH_ROOT}/modified.tsv"

  while IFS=$'\t' read -r _ relative; do
    source="${PATCH_ROOT}/payload/additions/${relative}"
    target="${TARGET_ROOT}/${relative}"
    mkdir -p "$(dirname "${target}")"
    cp -p "${source}" "${target}"
    echo "added: ${relative}"
  done < "${PATCH_ROOT}/added.tsv"
}

require_source_tree
validate_bundle
validate_target
apply_payload
"${PATCH_ROOT}/verify.sh" "${TARGET_ROOT}"

echo "LightX2V MiniMax-H3 4-step Ascend training patch: PASS"
