"""Turns the CI check screenshots into App Store screenshots and a manifest.

    python scripts/ci/appstore-shots.py <checks dir> <out dir>

Every picture is a capture of the real app running a real run (the synthetic sphere, the
project's own data): nothing is drawn, composited onto a device frame or retouched.

- iPhone and iPad: the simulator's own screenshots, kept as they are when they already have
  a size App Store Connect accepts (6.9"/6.7" iPhone, 13" iPad).
- Mac: the app's window, scaled to fit 2880x1800 (the largest Mac size) and centred on the
  app's own background colour, because a window capture has whatever size the window had.

Writes <out>/<display type>/<nn>-<screen>.png and <out>/manifest.json, and fails when a
picture has a size Apple does not take or a screen is missing.
"""

import json
import sys
from pathlib import Path

from PIL import Image

SCREENS = ["run", "surface", "sheets", "runs", "new-run"]
# Display types of App Store Connect and the sizes each accepts (portrait first).
IPHONE = {"APP_IPHONE_67": [(1320, 2868), (1290, 2796)]}
IPAD = {"APP_IPAD_PRO_3GEN_129": [(2064, 2752), (2048, 2732)]}
MAC_SIZE = (2880, 1800)
MAC_BACKGROUND = (0x16, 0x1A, 0x21)

checks, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = []
problems = []


def place(kind, sizes):
    # The device is recognised by the size of its screenshots, whatever the simulator was called.
    found = [path for path in sorted(checks.glob("*-surface.png")) if not path.name.startswith("mac-") and Image.open(path).size in sizes]
    if not found:
        problems.append(f"no {kind} screenshots (sizes {sizes}) in {checks}")
        return
    device = found[0].name[: -len("-surface.png")]
    for number, screen in enumerate(SCREENS, 1):
        source = checks / f"{device}-{screen}.png"
        if not source.exists():
            problems.append(f"missing {source.name}")
            continue
        image = Image.open(source).convert("RGB")
        if image.size not in sizes:
            problems.append(f"{source.name}: {image.size[0]}x{image.size[1]} is not a size {kind} accepts ({sizes})")
            continue
        target = out / kind / f"{number:02d}-{screen}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, optimize=True)
        rows.append({"displayType": kind, "file": f"{kind}/{target.name}", "pixels": f"{image.size[0]}x{image.size[1]}", "source": source.name})


for kind, sizes in {**IPHONE, **IPAD}.items():
    place(kind, sizes)

for number, screen in enumerate(SCREENS, 1):
    source = checks / f"mac-{screen}.png"
    if not source.exists():
        problems.append(f"missing {source.name}")
        continue
    window = Image.open(source).convert("RGB")
    scale = min(MAC_SIZE[0] * 0.94 / window.width, MAC_SIZE[1] * 0.94 / window.height, 1.0)
    window = window.resize((round(window.width * scale), round(window.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", MAC_SIZE, MAC_BACKGROUND)
    canvas.paste(window, ((MAC_SIZE[0] - window.width) // 2, (MAC_SIZE[1] - window.height) // 2))
    target = out / "APP_DESKTOP" / f"{number:02d}-{screen}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, optimize=True)
    rows.append({"displayType": "APP_DESKTOP", "file": f"APP_DESKTOP/{target.name}", "pixels": f"{MAC_SIZE[0]}x{MAC_SIZE[1]}", "source": source.name})

(out / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    print(f"{row['displayType']:24} {row['pixels']:10} {row['file']}  <- {row['source']}")
if problems:
    print("Problems:\n  " + "\n  ".join(problems))
    sys.exit(1)
