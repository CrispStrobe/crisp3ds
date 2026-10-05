"""Lists the files of the dense pipeline's source archive, one per line.

Everything under scripts/turntable_mesh, plus whatever other modules under scripts/
those import (followed transitively through the Python sources, so a new import
elsewhere in scripts/ is picked up without editing a list), the engine contract, the
replay fixture and the license. Only files tracked by git are listed.

Run from the repository root:  python apps/studio/scripts/pipeline-files.py
"""

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
ALWAYS = ["LICENSE", "docs/ENGINE-CONTRACT.md", "scripts/__init__.py"]
TREES = ["scripts/turntable_mesh", "tests/fixtures/dense-run-sphere"]


def tracked(*paths):
    out = subprocess.run(["git", "ls-files", "-z", "--", *paths], check=True, capture_output=True, text=True).stdout
    return [name for name in out.split("\0") if name]


def module_file(module):
    """scripts.a.b -> scripts/a/b.py or scripts/a/b/__init__.py, if it exists."""
    base = ROOT / Path(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def imports(path):
    """Modules under scripts/ that a source file imports, absolute or relative."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    package = Path(path).parent.as_posix().replace("/", ".")
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return {name for name in found if name == "scripts" or name.startswith("scripts.")}


def main():
    files = set(tracked(*ALWAYS, *TREES))
    queue = [name for name in files if name.endswith(".py")]
    while queue:
        for module in imports(queue.pop()):
            # The module itself and the __init__.py of every package above it.
            parts = module.split(".")
            for depth in range(1, len(parts) + 1):
                name = module_file(".".join(parts[:depth]))
                if name and name not in files and tracked(name):
                    files.add(name)
                    queue.append(name)
    missing = [name for name in ALWAYS if name not in files]
    if missing:
        sys.exit("not tracked by git: " + ", ".join(missing))
    print("\n".join(sorted(files)))


if __name__ == "__main__":
    main()
