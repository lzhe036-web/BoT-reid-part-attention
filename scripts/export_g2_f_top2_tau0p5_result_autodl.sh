#!/usr/bin/env bash
# Verify a completed G2-F package, then emit an AutoDL-downloadable archive.
set -euo pipefail

OUTPUT_DIR="${1:?usage: $0 OUTPUT_DIR}"
PACKAGE_DIR="${G2_F_PACKAGE_DIR:-${OUTPUT_DIR}/g2_f_top2_tau0p5_result_package}"
ARCHIVE="${G2_F_ARCHIVE:-/root/autodl-tmp/exports/$(basename "${PACKAGE_DIR}").tar.gz}"

test -d "${PACKAGE_DIR}" || { echo "Delivery package is absent or incomplete." >&2; exit 1; }
test -f "${PACKAGE_DIR}/SHA256SUMS" || { echo "Missing package SHA256SUMS." >&2; exit 1; }
(cd "${PACKAGE_DIR}" && sha256sum -c SHA256SUMS)
mkdir -p "$(dirname "${ARCHIVE}")"
test ! -e "${ARCHIVE}" || { echo "Refusing to overwrite archive: ${ARCHIVE}" >&2; exit 1; }
tar -C "$(dirname "${PACKAGE_DIR}")" -czf "${ARCHIVE}" "$(basename "${PACKAGE_DIR}")"
(cd "$(dirname "${ARCHIVE}")" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
printf 'archive=%s\nsha256=%s.sha256\n' "${ARCHIVE}" "${ARCHIVE}"
