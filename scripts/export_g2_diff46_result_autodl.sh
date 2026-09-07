#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/root/autodl-tmp/experiments/BoT/c2_l03_mgdg_g2_d1_tau0p5_seed42_market1501}"
PACKAGE_DIR="${G2_D1_PACKAGE_DIR:-${OUTPUT_DIR}/g2_d1_result_package}"
EXPORT_DIR="/root/autodl-tmp/exports"
ARCHIVE="${G2_D1_ARCHIVE:-${EXPORT_DIR}/$(basename "${PACKAGE_DIR}").tar.gz}"

test -f "${OUTPUT_DIR}/g2_d1_formal_result.json" || { echo "Formal result is absent." >&2; exit 1; }
test -s "${PACKAGE_DIR}/SHA256SUMS" || { echo "Delivery package is absent or incomplete." >&2; exit 1; }
test ! -e "${ARCHIVE}" && test ! -e "${ARCHIVE}.sha256" || { echo "Refusing to overwrite export." >&2; exit 1; }
mkdir -p "${EXPORT_DIR}"
(cd "${PACKAGE_DIR}" && sha256sum -c SHA256SUMS)
tar -C "$(dirname "${PACKAGE_DIR}")" -czf "${ARCHIVE}" "$(basename "${PACKAGE_DIR}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
printf 'archive=%s\nsha256=%s\n' "${ARCHIVE}" "${ARCHIVE}.sha256"
