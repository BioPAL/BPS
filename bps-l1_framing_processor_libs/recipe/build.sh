
#!/usr/bin/env bash
set -euo pipefail

# Install shared libraries in an isolated directory.
LIBDIR="${PREFIX}/lib/bps-l1-framing-processor-libs"
mkdir -p "${LIBDIR}"

if [[ ! -d "${SRC_DIR}" ]]; then
    echo "ERROR: Source library directory not found: ${SRC_DIR}" >&2
    exit 1
fi

# Copy shared libraries only, preserving symlinks.
find "${SRC_DIR}" -maxdepth 1 \
    \( -type f -o -type l \) \
    \( -name '*.so' -o -name '*.so.*' \) \
    -exec cp -a -t "${LIBDIR}" {} +

# Ensure at least one shared library was copied.
if ! find "${LIBDIR}" -maxdepth 1 \
    \( -name '*.so' -o -name '*.so.*' \) \
    -print -quit | grep -q .; then
    echo "ERROR: No shared libraries found in ${LIBDIR}" >&2
    exit 1
fi

# patchelf is provided by the build environment.
if ! command -v patchelf >/dev/null 2>&1; then
    echo "ERROR: patchelf is required to set RUNPATH." >&2
    exit 1
fi

# Let each shared library find sibling libraries.
while IFS= read -r -d '' lib; do
    if [[ -f "${lib}" ]] && file "${lib}" | grep -q ELF; then
        echo "Setting RUNPATH: ${lib}"
        patchelf --set-rpath '$ORIGIN' "${lib}"
    fi
done < <(
    find "${LIBDIR}" -maxdepth 1 \
        \( -type f -o -type l \) \
        \( -name '*.so' -o -name '*.so.*' \) \
        -print0
)

echo "Packaged shared libraries:"
find "${LIBDIR}" -maxdepth 1 \
    \( -name '*.so' -o -name '*.so.*' \) -printf '%f\n'

mkdir -p "${PREFIX}/etc/conda/activate.d" \
         "${PREFIX}/etc/conda/deactivate.d"

cat > "${PREFIX}/etc/conda/activate.d/bps-l1-framing-processor-libs.sh" <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/bps-l1-framing-processor-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

cat > "${PREFIX}/etc/conda/deactivate.d/bps-l1-framing-processor-libs.sh" <<'EOF'
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -Fvx "${CONDA_PREFIX}/lib/bps-l1-framing-processor-libs" | paste -sd:)"
EOF