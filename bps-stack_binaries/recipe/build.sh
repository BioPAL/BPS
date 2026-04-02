#!/bin/bash
set -euo pipefail

echo "PREFIX = $PREFIX"

mkdir -p "$PREFIX/bin"
mkdir -p "$PREFIX/lib"

echo "Copying binaries..."
cp "$SRC_DIR/bin/"* "$PREFIX/bin/"

echo "Copying shared libraries..."
cp "$SRC_DIR/lib/"*.so* "$PREFIX/lib/"


chmod +x "$PREFIX/bin/"*

echo "Patching RPATH for binary..."
bin=$PREFIX/bin/BPSStackProcessor
echo "  -> $bin"
patchelf --set-rpath "\$ORIGIN/../lib" "$bin"

echo "Patching RPATH for shared libraries..."
for lib in \
    "$PREFIX/lib/libareapplib_common.so" \
    "$PREFIX/lib/libare_parsing.so" \
    "$PREFIX/lib/libare_sarbase.so" \
    "$PREFIX/lib/libare_core.so" \
    "$PREFIX/lib/libare_focus.so"
do
    echo "  -> $lib"
    patchelf --set-rpath '$ORIGIN' "$lib"
done

echo "Verifying binaries..."
echo "Checking $bin"
ldd "$bin"

echo "Build completed successfully!"
