"""Versioned, hash-bound original-pixel SAM prompt recipes; no inference."""

import hashlib
import json
from pathlib import Path

from scripts.object_motion import sam_mask_trial as smoke
from scripts.object_motion import ycb_object_masks as common


WIDTH, HEIGHT = common.IMAGE_SIZE
CANONICAL_KEYS = ("name", "source_sha256", "box_xyxy_original_pixels", "points_xy_label")


def canonical_row(row, box):
    """The exact per-view prompt identity shared with downstream composition."""
    return {"name": row["name"], "source_sha256": row["source_sha256"],
            "box_xyxy_original_pixels": list(box), "points_xy_label": row["points_xy_label"]}


def row_sha256(row, box):
    payload = json.dumps(canonical_row(row, box), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_geometry(box, points, version):
    if version not in (1, 2):
        raise ValueError("unknown prompt recipe version")
    if (not isinstance(box, list) or len(box) != 4 or
            any(type(value) is not int for value in box)):
        raise ValueError("box must contain four integer original-pixel coordinates")
    left, top, right, bottom = box
    if not (0 <= left < right <= WIDTH and 0 <= top < bottom <= HEIGHT):
        raise ValueError("box outside original image or inverted")
    if (not isinstance(points, list) or
            (version == 1 and len(points) != 4) or
            (version == 2 and not 1 <= len(points) <= 16)):
        raise ValueError("recipe point count outside versioned contract")
    seen = set()
    positives = 0
    for index, point in enumerate(points):
        if (not isinstance(point, list) or len(point) != 3 or
                any(type(value) is not int for value in point)):
            raise ValueError("point must be integer [x,y,label]")
        x, y, label = point
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT and label in (0, 1)):
            raise ValueError("point outside original image or invalid label")
        if version == 1 and label != (1 if index == 0 else 0):
            raise ValueError("v1 ordered one-positive/three-negative label changed")
        if (x, y) in seen:
            raise ValueError("duplicate point")
        seen.add((x, y))
        positives += label
        if label == 1 and not (left <= x < right and top <= y < bottom):
            raise ValueError("positive point must be inside box")
        if version == 1 and not (left <= x < right and top <= y < bottom):
            raise ValueError("v1 points must all be inside box")
    if positives == 0:
        raise ValueError("at least one positive point required")
    return True


def load_v2(path, expected_sha, training_rows, *, expected_names=None):
    """A caller must supply a separately reviewed exact recipe SHA-256."""
    path = Path(path)
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
            any(character not in "0123456789abcdef" for character in expected_sha)):
        raise ValueError("expected reviewed recipe SHA-256 must be lowercase hex")
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 or
            common.digest(path) != expected_sha):
        raise ValueError("recipe missing, linked, oversize or not approved hash")
    recipe = json.loads(path.read_text())
    if recipe.get("schema") != "sam21_m1_prompt_recipe_v2":
        raise ValueError("unsupported recipe version")
    box = recipe.get("box_xyxy_original_pixels")
    selected = recipe.get("selected_training_names")
    images = recipe.get("images")
    if (not isinstance(selected, list) or not selected or len(selected) > 8 or
            not all(isinstance(name, str) for name in selected) or
            len(set(selected)) != len(selected) or
            not isinstance(images, list) or len(images) != len(selected) or
            not all(isinstance(item, dict) for item in images) or
            [item.get("name") for item in images] != selected or
            (expected_names is not None and selected != list(expected_names))):
        raise ValueError("recipe selected TRAIN names missing, extra or reordered")
    train = {Path(row["path"]).name: row for row in training_rows}
    if len(train) != len(training_rows) or any(name not in train for name in selected):
        raise ValueError("recipe contains nontraining/duplicate photo names")
    selected_rows = [train[name] for name in selected]
    for item, row in zip(images, selected_rows):
        if item.get("source_sha256") != row["sha256"]:
            raise ValueError("recipe TRAIN photo SHA differs")
        validate_geometry(box, item.get("points_xy_label"), 2)
    if selected != [Path(row["path"]).name for row in training_rows if Path(row["path"]).name in selected]:
        raise ValueError("recipe not in source package order")
    return recipe, selected_rows


def v1_adapter(path, expected_sha, training_rows):
    """Read the frozen board5 v1 fixture as canonical rows without rewriting it."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or common.digest(path) != expected_sha):
        raise ValueError("v1 fixture differs from its frozen bytes")
    fixture = json.loads(path.read_text())
    if fixture.get("schema") != "sam21_mustard_m1_board_negative5_prompts_v1":
        raise ValueError("not the frozen board5 v1 schema")
    box = fixture.get("box_xyxy_original_pixels")
    lookup = {Path(row["path"]).name: row for row in training_rows}
    if [item.get("name") for item in fixture.get("images", [])] != [
            "NP3_318.jpg", "NP3_330.jpg", "NP3_336.jpg", "NP3_342.jpg", "NP3_348.jpg"]:
        raise ValueError("v1 five-view inventory differs")
    canonical = []
    for item in fixture["images"]:
        if item["name"] not in lookup or item.get("source_sha256") != lookup[item["name"]]["sha256"]:
            raise ValueError("v1 source photo differs")
        validate_geometry(box, item.get("points_xy_label"), 1)
        canonical.append({**canonical_row(item, box), "recipe_row_sha256": row_sha256(item, box)})
    return canonical
