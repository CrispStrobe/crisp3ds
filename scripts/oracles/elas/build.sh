#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
oracle_dir="$repo_dir/.local-tools/oracles/elas"
source_dir="$oracle_dir/upstream"
build_dir="$oracle_dir/build"
commit=862ee4ef30a070753e9b92965855562a4482da05
archive_sha256=ae32ab63888994910e6813b65242a8f05ffc86e448048d6fe67dcd1eb4830a8d

free_kib=$(df -Pk "$repo_dir" | awk 'NR==2 {print $4}')
if [ "$free_kib" -lt 11534336 ]; then
  echo "Need at least 11 GiB free before libELAS checkout/build" >&2
  exit 1
fi
mkdir -p "$oracle_dir" "$build_dir" "$repo_dir/.local-tools/tmp"
if [ ! -d "$source_dir/.git" ]; then
  git clone --depth 1 https://github.com/kou1okada/libelas.git "$source_dir"
fi
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]; then
  git -C "$source_dir" fetch --depth 1 origin "$commit"
  git -C "$source_dir" checkout --detach "$commit"
fi
if [ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]; then
  echo "libELAS commit mismatch" >&2
  exit 1
fi
if [ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]; then
  echo "libELAS source checkout must be clean" >&2
  exit 1
fi
actual_archive_sha256=$(git -C "$source_dir" archive --format=tar HEAD | shasum -a 256 | awk '{print $1}')
if [ "$actual_archive_sha256" != "$archive_sha256" ]; then
  echo "libELAS source archive hash mismatch" >&2
  exit 1
fi

TMPDIR="$repo_dir/.local-tools/tmp" /usr/bin/clang++ -arch x86_64 -msse3 -std=c++17 -O2 -DNDEBUG -Wno-c++11-narrowing \
  -I"$source_dir/src" \
  "$repo_dir/scripts/oracles/elas/oracle.cpp" \
  "$source_dir/src/elas.cpp" "$source_dir/src/descriptor.cpp" \
  "$source_dir/src/filter.cpp" "$source_dir/src/matrix.cpp" \
  "$source_dir/src/triangle.cpp" -o "$build_dir/elas_oracle"
python3 "$repo_dir/scripts/oracles/elas/record_build.py" "$source_dir" "$build_dir/elas_oracle" "$build_dir/provenance.json"
echo "$build_dir/elas_oracle"
