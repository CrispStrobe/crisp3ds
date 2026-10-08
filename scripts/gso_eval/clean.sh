#!/bin/bash
# Deletes a case's renders and run intermediates; keeps mesh, scores, sheets and reports.
here="${GSO_EVAL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/.local-tools/gso-eval}"
work=$here/work/$1
[ -d "$work/run" ] || [ -d "$work/capture" ] || exit 0
keep=$work/keep
mkdir -p "$keep"
for f in run/mesh/mesh.stl run/frontend/frontend.json run/frontend/mask-contact-sheet.png run/frontend/sparse-overlay.png \
         run/input-sheet.png run/pipeline.json run/check/preview.png run/check/photo-overlay.png run/check/result.json \
         capture/truth.json capture/lens.json; do
  [ -f "$work/$f" ] && cp "$work/$f" "$keep/$(echo "$f" | tr '/' '_')"
done
rm -rf "$work/run" "$work/capture/photos" "$work/capture/masks"
