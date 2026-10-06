"""Rewrites src-tauri/icons/ios/*.png without an alpha channel.

`tauri icon --ios-color` fills the transparent corners but still writes RGBA files, and
App Store Connect refuses an app icon that has an alpha channel. Run after `tauri icon`:

    npx tauri icon ../desktop/src-tauri/icons/source.svg --ios-color "#142630"
    python scripts/ios-icons-opaque.py

Needs Pillow.
"""

from pathlib import Path

from PIL import Image

BACKGROUND = (0x14, 0x26, 0x30)
folder = Path(__file__).resolve().parent.parent / "src-tauri" / "icons" / "ios"
changed = 0
for path in sorted(folder.glob("*.png")):
    image = Image.open(path)
    if image.mode == "RGB":
        continue
    flat = Image.new("RGB", image.size, BACKGROUND)
    flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[3])
    flat.save(path, optimize=True)
    changed += 1
print(f"{changed} icons rewritten without alpha in {folder}")
