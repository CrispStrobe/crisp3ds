#!/bin/bash
# Runs pipeline.sh for each "CASE MODEL [render options]" line on stdin, one at a time.
scripts="$(cd "$(dirname "$0")" && pwd)"
here="${GSO_EVAL_DIR:-$(cd "$scripts/../.." && pwd)/.local-tools/gso-eval}"
while read -r case_name model rest; do
  [ -z "$case_name" ] && continue
  [ -f "$here/work/$case_name/score/result.json" ] && continue
  echo "start $case_name $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  bash "$scripts/pipeline.sh" "$case_name" "$model" $rest < /dev/null
  echo "end $case_name $(date +%H:%M:%S) $(grep -E 'F1@0.005' "$here/work/$case_name/log.txt" | head -1 | sed 's/.*| F1/F1/')"
done
