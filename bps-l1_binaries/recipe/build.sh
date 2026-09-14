#!/bin/bash

# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "PREFIX = $PREFIX"

mkdir -p "$PREFIX/bin"
mkdir -p "$PREFIX/lib"

echo "Copying binaries..."
cp "$SRC_DIR/bin/"* "$PREFIX/bin/"

echo "Copying shared libraries..."
cp "$SRC_DIR/lib/"*.so* "$PREFIX/lib/"


chmod +x "$PREFIX/bin/"*

echo "Patching RPATH for binaries..."
for bin in \
    "$PREFIX/bin/BiomassL0ImportPreProc" \
    "$PREFIX/bin/BPSL1CoreProcessor" \
    "$PREFIX/bin/IonosphericHeightIRI20Estimator"
do
    echo "  -> $bin"
    patchelf --set-rpath "\$ORIGIN/../lib" "$bin"
done

echo "Patching RPATH for shared libraries..."
for lib in \
    "$PREFIX/lib/libareapplib_common.so" \
    "$PREFIX/lib/libare_parsing.so" \
    "$PREFIX/lib/libare_sarbase.so" \
    "$PREFIX/lib/libare_core.so" \
    "$PREFIX/lib/libareapplib_ionosphericCalibration.so" \
    "$PREFIX/lib/libareapplib_rfimitigator.so" \
    "$PREFIX/lib/libare_focus.so"
do
    echo "  -> $lib"
    patchelf --set-rpath '$ORIGIN' "$lib"
done

echo "Verifying binaries..."
for bin in \
    "$PREFIX/bin/BiomassL0ImportPreProc" \
    "$PREFIX/bin/BPSL1CoreProcessor" \
    "$PREFIX/bin/IonosphericHeightIRI20Estimator"
do
    echo "Checking $bin"
    ldd "$bin"
done

echo "Build completed successfully!"
