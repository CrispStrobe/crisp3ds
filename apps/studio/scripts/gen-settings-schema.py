"""Writes apps/studio/src-tauri/settings-schema.json from the Python reference.

The native crate has the settings (DenseConfig) but not their group and meaning,
which the generated settings form needs. Until the crate offers a schema itself,
the shell embeds this file. A Rust test checks that its names and defaults are
the crate's own and those of tests/fixtures/dense-config-defaults.json.

Run from the repository root:  python apps/studio/scripts/gen-settings-schema.py
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from scripts.turntable_mesh.dense_config import settings_schema  # noqa: E402

target = Path("apps/studio/src-tauri/settings-schema.json")
target.write_text(json.dumps({"settings": settings_schema()}, indent=2) + "\n", encoding="utf-8")
print(f"{target}: {len(settings_schema())} settings")
