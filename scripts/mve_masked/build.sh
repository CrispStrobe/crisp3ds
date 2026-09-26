#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMMIT=bf2279f161ba962072ecac85224c15e82bc5f52e
SOURCE="$ROOT/.local-tools/mve-spike/mve-$COMMIT"
BUILD_ROOT="${1:-$ROOT/.local-tools/mve-masked}"
DEST="$BUILD_ROOT/mve-$COMMIT"
LOCAL_TMP="$ROOT/.local-tools/tmp"
BUILD_PARENT="$(dirname "$BUILD_ROOT")"

if [[ ! -d "$SOURCE" ]]; then
  echo "Pinned selected MVE source is missing: $SOURCE" >&2
  exit 1
fi
if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing source: $DEST" >&2
  exit 1
fi
if [[ ! -d "$BUILD_PARENT" ]]; then
  echo "Build root parent must already exist: $BUILD_PARENT" >&2
  exit 1
fi

SOURCE_KIB="$(du -sk "$SOURCE" | awk '{print $1}')"
FREE_KIB="$(df -Pk "$BUILD_PARENT" | awk 'NR == 2 {print $4}')"
REQUIRED_KIB=$((10 * 1024 * 1024 + SOURCE_KIB + 100 * 1024))
if [[ "$FREE_KIB" -lt "$REQUIRED_KIB" ]]; then
  echo "Need a 10 GiB free-space reserve plus source copy and build allowance" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
mkdir -p "$LOCAL_TMP"
export TMPDIR="$LOCAL_TMP"
cp -R "$SOURCE" "$DEST"
for component in libs/math libs/util libs/mve libs/dmrecon apps/dmrecon; do
  make -C "$DEST/$component" clean
done
patch --batch -p1 -d "$DEST" -i "$ROOT/scripts/mve_masked/mve-object-mask.patch"

for component in libs/math libs/util libs/mve libs/dmrecon apps/dmrecon; do
  make -C "$DEST/$component" -j2 CXX=clang++
done

echo "$DEST/apps/dmrecon/dmrecon"
