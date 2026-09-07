#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_c_hidden256_tau0p5_seed42_market1501}"
PACKAGE_DIR="${OUTPUT_DIR}/g2_c_hidden256_tau0p5_result_package"
EXPORT_DIR="/root/autodl-tmp/exports"
ARCHIVE="${EXPORT_DIR}/g2_c_hidden256_tau0p5_result_package.tar.gz"

if [[ ! -f "${OUTPUT_DIR}/g2_c_hidden256_tau0p5_formal_result.json" ]]; then
  printf 'Formal G2-C result is absent: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PACKAGE_DIR}" || ! -s "${PACKAGE_DIR}/SHA256SUMS.txt" ]]; then
  printf 'Formal G2-C delivery package is absent or incomplete: %s\n' "${PACKAGE_DIR}" >&2
  exit 1
fi
if [[ -e "${ARCHIVE}" || -e "${ARCHIVE}.sha256" ]]; then
  printf 'Refusing to overwrite export: %s\n' "${ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${EXPORT_DIR}"
(cd "${OUTPUT_DIR}" && sha256sum -c "${PACKAGE_DIR}/SHA256SUMS.txt")
tar -C "${OUTPUT_DIR}" -czf "${ARCHIVE}" "$(basename "${PACKAGE_DIR}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
printf 'archive=%s\nsha256=%s\n' "${ARCHIVE}" "${ARCHIVE}.sha256"
