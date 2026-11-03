#!/bin/sh
# Fetches the NVIDIA DLSS SDK static loader and Linux runtime libraries, which stay out of git.
# The headers in include/ come from the same release; see LICENSE.txt for the NVIDIA RTX SDK terms.
set -e
VER="${VER:-v310.9.16}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$HERE/../../bin/amd64}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q --depth 1 --filter=blob:none --sparse --branch "$VER" https://github.com/NVIDIA/DLSS.git "$TMP/dlss"
git -C "$TMP/dlss" sparse-checkout set --no-cone '/lib/Linux_x86_64/libnvsdk_ngx.a' '/lib/Linux_x86_64/rel/*'
mkdir -p "$HERE/lib" "$BIN"
cp "$TMP/dlss/lib/Linux_x86_64/libnvsdk_ngx.a" "$HERE/lib/"
cp "$TMP/dlss/lib/Linux_x86_64/rel/"*.so.* "$BIN/"
echo "NGX $VER: $HERE/lib/libnvsdk_ngx.a and runtime libraries in $BIN"
