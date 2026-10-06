"""Fails when a screenshot is blank: one colour, or nearly so.

    python scripts/ci/image-check.py shot.png [more.png ...]

A window that never rendered, a white web view or a black Metal layer all come out as a
picture with almost no variation. Each file must have a standard deviation of its grey
values above a threshold and enough distinct grey levels. Prints size and the two numbers.
Needs Pillow.
"""

import sys

from PIL import Image, ImageStat

MIN_STDDEV = 12.0
MIN_LEVELS = 24

failed = False
for path in sys.argv[1:]:
    image = Image.open(path)
    grey = image.convert("L")
    stddev = ImageStat.Stat(grey).stddev[0]
    levels = sum(1 for count in grey.histogram() if count > 0)
    blank = stddev < MIN_STDDEV or levels < MIN_LEVELS
    failed |= blank
    print(f"{'BLANK' if blank else 'ok   '} {path}: {image.size[0]}x{image.size[1]}, grey stddev {stddev:.1f}, {levels} grey levels")
sys.exit(1 if failed else 0)
