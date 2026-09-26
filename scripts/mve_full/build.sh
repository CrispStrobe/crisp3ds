#!/usr/bin/env bash
# Reproduce the selected MVE CPU tools from the pinned upstream source archive.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
revision='bf2279f161ba962072ecac85224c15e82bc5f52e'
archive="$repo_root/.local-tools/mve-spike/mve-bf2279f.tar.gz"
source_dir="$repo_root/.local-tools/mve-spike/mve-$revision"
patch_file="$repo_root/scripts/mve_full/sift_only.patch"
expected='0e33d40b150313617d28ab4297459939e9ad59c47f21566ed649daf3b34c47f6'

if [[ "$(uname -s)" != Darwin && "$(uname -s)" != Linux ]]; then
  echo 'Selected GNU Make build currently supports macOS and Linux only.' >&2
  exit 2
fi
if [[ ! -f "$archive" ]]; then
  echo "Pinned archive missing: $archive" >&2
  echo "Fetch https://github.com/simonfuhrmann/mve/archive/$revision.tar.gz into that path after checking disk budget." >&2
  exit 2
fi
actual="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$archive")"
if [[ "$actual" != "$expected" ]]; then
  echo 'Pinned MVE archive checksum mismatch.' >&2
  exit 2
fi
if [[ ! -d "$source_dir" ]]; then
  free_bytes="$(python3 -c 'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)' "$repo_root")"
  if (( free_bytes < 12 * 1024 * 1024 * 1024 )); then
    echo 'Insufficient free space for extraction plus 10 GiB reserve.' >&2
    exit 2
  fi
  tar -xzf "$archive" -C "$repo_root/.local-tools/mve-spike"
fi
if git -C "$source_dir" apply --unidiff-zero --reverse --check "$patch_file" 2>/dev/null; then
  : # Already patched.
else
  git -C "$source_dir" apply --unidiff-zero --check "$patch_file"
  git -C "$source_dir" apply --unidiff-zero "$patch_file"
fi
python3 "$repo_root/scripts/mve_full/verify_source.py" --archive "$archive" --source "$source_dir"
if [[ ! -d "$repo_root/.local-tools/tmp" ]]; then
  mkdir -p "$repo_root/.local-tools/tmp"
fi
export TMPDIR="$repo_root/.local-tools/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
jobs="${MVE_JOBS:-2}"
if ! [[ "$jobs" =~ ^[1-4]$ ]]; then
  echo 'MVE_JOBS must be an integer from 1 to 4.' >&2
  exit 2
fi
cxx="${CXX:-}"
if [[ -z "$cxx" ]]; then
  cxx="$(command -v clang++ || command -v g++)"
fi
for library in math util mve sfm dmrecon fssr; do
  make -C "$source_dir/libs/$library" -j "$jobs" CXX="$cxx"
done
for app in makescene sfmrecon dmrecon scene2pset fssrecon meshclean; do
  make -C "$source_dir/apps/$app" -j "$jobs" CXX="$cxx"
done
if ar -t "$source_dir/libs/sfm/libmve_sfm.a" | grep -qx 'surf.o'; then
  echo 'Rejected: selected SfM archive still contains surf.o.' >&2
  exit 2
fi
echo "Selected CPU binaries built under $source_dir/apps"
