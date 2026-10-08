#!/bin/bash
# Downloads Google Scanned Objects (CC BY 4.0) from Gazebo Fuel and unpacks each into models/NAME/.
here="${GSO_EVAL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/.local-tools/gso-eval}"
mkdir -p "$here/models" && cd "$here/models" || exit 1
for name in "$@"; do
  [ -d "$name" ] && continue
  curl -s -m 300 -L -o "$name.zip" "https://fuel.gazebosim.org/1.0/GoogleResearch/models/$name/1/$name.zip" || exit 1
  mkdir -p "$name" && unzip -q -o "$name.zip" -d "$name" && rm "$name.zip"
  echo "$name $(du -sh "$name" | cut -f1)"
done
