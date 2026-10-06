"""Optional photo-only SAM2.1 tiny segmentation on CPU or Apple MPS.

Frozen prompts use coarse support, never reference geometry. The selected
prompt-connected component has its holes filled: inspect these inferred object
silhouettes before reconstruction, especially objects with real through-holes.
Torch and SAM are imported only when inference is requested.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.classical_backend.geometry import digest

MAX_PIXELS = 12_000_000
MAX_VIEWS = 255
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def selected_images(images, views=0):
    if (
        isinstance(views, bool)
        or not isinstance(views, int)
        or not 0 <= views <= MAX_VIEWS
    ):
        raise ValueError("views must be an integer in 0..255; zero selects all")
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in SUFFIXES)
    if not paths or len(paths) > MAX_VIEWS:
        raise ValueError("segmentation input must contain 1..255 photos")
    if views and views < len(paths):
        paths = [
            paths[index]
            for index in np.linspace(0, len(paths), views, endpoint=False, dtype=int)
        ]
    return paths


def input_inventory(paths, masks):
    return [
        {
            "name": path.name,
            "rgb_sha256": digest(path),
            "coarse_mask_sha256": digest(masks / (path.name + ".png")),
        }
        for path in paths
    ]


def model_inventory(source, checkpoint, config):
    configuration = source / "sam2" / config
    if not configuration.is_file():
        raise ValueError("SAM model configuration must exist under source/sam2")
    paths = sorted(
        p
        for p in source.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
    )
    if not paths or not (source / "sam2/build_sam.py").is_file():
        raise ValueError("source must be a SAM2 checkout")
    return {
        "checkpoint_sha256": digest(checkpoint),
        "configuration_sha256": digest(configuration),
        "source_files_sha256": {
            str(path.relative_to(source)): digest(path) for path in paths
        },
    }


def interior_positive_points(support, distance, primary):
    """Sample conservative quadrant interiors; never infer missing support."""
    yy, xx = np.nonzero(support)
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    clearance = max(2.0, 0.25 * float(distance.max()))
    separation = max(2.0, 0.2 * min(x1 - x0, y1 - y0))
    points = [primary.copy()]
    for left, top, right, bottom in (
        (x0, y0, mx, my),
        (mx, y0, x1, my),
        (x0, my, mx, y1),
        (mx, my, x1, y1),
    ):
        region = distance[top:bottom, left:right]
        if not region.size or region.max() < clearance:
            continue
        yy, xx = np.ogrid[top:bottom, left:right]
        separated = (xx - primary[0]) ** 2 + (yy - primary[1]) ** 2 >= separation**2
        score = np.where(separated & (region >= clearance), region, 0)
        if not score.any():
            continue
        y, x = np.unravel_index(score.argmax(), score.shape)
        point = [int(x + left), int(y + top)]
        if point not in points:
            points.append(point)
    return points, clearance


def exterior_negative_points(box, dimensions):
    """Use an explicit expanded-box containment prior, never mask-hole negatives."""
    width, height = dimensions
    x0, y0, x1, y1 = box
    margin = max(24, int(np.ceil(0.2 * max(x1 - x0, y1 - y0))))
    left, top, right, bottom = (
        x0 - margin - 1,
        y0 - margin - 1,
        x1 + margin,
        y1 + margin,
    )
    mx, my = (x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2
    candidates = [
        [mx, top],
        [right, my],
        [mx, bottom],
        [left, my],
        [left, top],
        [right, top],
        [left, bottom],
        [right, bottom],
    ]
    return [
        point
        for point in candidates
        if 0 <= point[0] < width and 0 <= point[1] < height
    ], margin


def foreground_distance(support, bounds):
    """Exact full-frame EDT using the foreground box and its real zero halo.

    All foreground is inside bounds. Background distances outside the box are
    zero, and the first surrounding real zero pixels already contain every
    possible nearest-background site. Clamp to the image rather than inventing
    an exterior zero border when foreground touches an image edge.
    """
    x0, y0, x1, y1 = bounds
    x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
    x1, y1 = min(support.shape[1], x1 + 1), min(support.shape[0], y1 + 1)
    distance = np.zeros(support.shape, dtype=np.float64)
    distance[y0:y1, x0:x1] = ndimage.distance_transform_edt(support[y0:y1, x0:x1])
    return distance


def frozen_prompts(paths, masks, automatic_cues=False):
    bounds, prompts, dimensions = [], [], None
    for path in paths:
        with Image.open(path) as photo:
            if photo.getexif().get(274, 1) != 1:
                raise ValueError("segmentation requires upright RGB pixels")
            width, height = photo.size
        if min(width, height) < 64 or width * height > MAX_PIXELS:
            raise ValueError("source image dimensions exceed segmentation bounds")
        if dimensions is not None and dimensions != (width, height):
            raise ValueError("common prompt box requires equal image dimensions")
        dimensions = (width, height)
        with Image.open(masks / (path.name + ".png")) as image:
            if (
                image.mode != "L"
                or image.size != dimensions
                or image.getexif().get(274, 1) != 1
            ):
                raise ValueError(
                    "coarse mask must be upright grayscale8 matching RGB dimensions"
                )
            support = np.array(image)
        if not np.isin(support, (0, 255)).all() or not support.any() or np.all(support):
            raise ValueError("coarse mask must be binary with both classes")
        yy, xx = np.nonzero(support)
        bounds.append(
            [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        )
        distance = foreground_distance(support > 0, bounds[-1])
        y, x = np.unravel_index(distance.argmax(), support.shape)
        prompts.append(
            {"name": path.name, "point_xy": [int(x), int(y)], "point_label": 1}
        )
        if automatic_cues:
            positives, clearance = interior_positive_points(
                support, distance, [int(x), int(y)]
            )
            prompts[-1].update(
                points_xy=positives,
                point_labels=[1] * len(positives),
                automatic_positive_clearance_pixels=clearance,
            )
    box = [
        min(row[0] for row in bounds),
        min(row[1] for row in bounds),
        max(row[2] for row in bounds),
        max(row[3] for row in bounds),
    ]
    for prompt in prompts:
        prompt["box_xyxy"] = box.copy()
        if automatic_cues:
            negatives, margin = exterior_negative_points(box, dimensions)
            prompt["points_xy"].extend(negatives)
            prompt["point_labels"].extend([0] * len(negatives))
            prompt["automatic_negative_margin_pixels"] = margin
    return prompts, dimensions


def clean_prediction(masks, scores, point, dimensions, *, preserve_holes=False):
    if not isinstance(preserve_holes, bool):
        raise ValueError("preserve_holes must be an explicit boolean")
    values, scores = np.asarray(masks), np.asarray(scores, dtype=float).reshape(-1)
    width, height = dimensions
    if (
        values.ndim != 3
        or values.shape[1:] != (height, width)
        or len(values) != len(scores)
        or not len(scores)
        or not np.isfinite(values).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("SAM returned malformed masks or scores")
    x, y = point
    candidates = [index for index, mask in enumerate(values) if mask[y, x] > 0]
    if not candidates:
        raise ValueError("SAM prediction does not contain its positive prompt")
    selected = max(candidates, key=lambda index: (scores[index], -index))
    raw = values[selected] > 0
    components, count = ndimage.label(raw)
    connected = components == components[y, x]
    clean = connected.copy() if preserve_holes else ndimage.binary_fill_holes(connected)
    if np.all(clean):
        raise ValueError("SAM returned an unbounded all-foreground silhouette")
    return (
        raw,
        clean,
        {
            "predicted_iou": float(scores[selected]),
            "raw_components": count,
            "raw_foreground_pixels": int(raw.sum()),
            "foreground_pixels": int(clean.sum()),
            "holes_filled_pixels": int(clean.sum() - connected.sum()),
        },
    )


def apply_prompt_overrides(prompts, path, inventory, dimensions):
    """Accept explicit photo-hash-bound point corrections; retain frozen boxes."""
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or data.get("schema") != "photo_only_segmentation_prompts_v1"
        or not isinstance(data.get("images"), list)
    ):
        raise ValueError("invalid photo-only segmentation prompt schema")
    hashes = {row["name"]: row["rgb_sha256"] for row in inventory}
    by_name = {prompt["name"]: prompt for prompt in prompts}
    seen = set()
    width, height = dimensions
    for row in data["images"]:
        if not isinstance(row, dict) or set(row) != {
            "name",
            "rgb_sha256",
            "points_xy",
            "point_labels",
        }:
            raise ValueError(
                "each prompt override requires name, RGB hash, points and labels"
            )
        name = row["name"]
        if not isinstance(name, str) or name not in by_name or name in seen:
            raise ValueError(
                "prompt override names must be unique selected source photos"
            )
        seen.add(name)
        if row["rgb_sha256"] != hashes[name]:
            raise ValueError("prompt override RGB hash does not match its source photo")
        points = np.asarray(row["points_xy"])
        labels = row["point_labels"]
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not 1 <= len(points) <= 32
            or points.dtype.kind not in "iuf"
            or not np.isfinite(points).all()
            or not (points == np.floor(points)).all()
            or not isinstance(labels, list)
            or len(labels) != len(points)
            or any(type(value) is not int or value not in (0, 1) for value in labels)
            or 1 not in labels
        ):
            raise ValueError(
                "prompt overrides require 1..32 finite integer XY points and binary labels with a positive point"
            )
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            raise ValueError("prompt override points are outside source image bounds")
        by_name[name]["points_xy"] = points.astype(int).tolist()
        by_name[name]["point_labels"] = labels.copy()
        by_name[name]["point_xy"] = points[labels.index(1)].astype(int).tolist()
        by_name[name]["explicit_photo_points"] = True
        if "automatic_negative_margin_pixels" in by_name[name]:
            by_name[name]["automatic_cues_replaced_by_explicit_points"] = True
    return prompts


def point_membership(mask, prompt):
    points = prompt.get("points_xy", [prompt["point_xy"]])
    labels = prompt.get("point_labels", [1])
    return [
        {
            "point_xy": [x, y],
            "label": expected,
            "included": bool(mask[y, x]),
            "passed": bool(mask[y, x]) == bool(expected),
        }
        for (x, y), expected in zip(points, labels)
    ]


def verify_cleaned_points(mask, prompt):
    for row in point_membership(mask, prompt):
        if not row["passed"]:
            raise ValueError(
                "cleaned silhouette violates an explicit positive/negative photo point"
            )


def select_prediction(masks, scores, prompt, dimensions, *, preserve_holes=False):
    """Select by photo cues after cleanup, then by model score; never geometry."""
    if not isinstance(preserve_holes, bool):
        raise ValueError("preserve_holes must be an explicit boolean")
    values, scores = np.asarray(masks), np.asarray(scores, dtype=float).reshape(-1)
    width, height = dimensions
    if (
        values.ndim != 3
        or values.shape[1:] != (height, width)
        or not len(scores)
        or len(values) != len(scores)
        or not np.isfinite(values).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("SAM returned malformed masks or scores")
    candidates, cleaned = [], {}
    for index, mask in enumerate(values):
        raw = mask > 0
        row = {
            "index": index,
            "predicted_iou": float(scores[index]),
            "passed": False,
            "raw_point_membership": point_membership(raw, prompt),
        }
        try:
            raw, clean, metrics = clean_prediction(
                values[index : index + 1],
                scores[index : index + 1],
                prompt["point_xy"],
                dimensions,
                preserve_holes=preserve_holes,
            )
            cleaned[index] = (raw, clean, metrics)
            row["clean_point_membership"] = point_membership(clean, prompt)
            row["passed"] = all(
                point["passed"] for point in row["clean_point_membership"]
            )
            if not row["passed"]:
                row["reason"] = "cleaned silhouette violates photographic point cues"
        except ValueError as failure:
            row["reason"] = str(failure)
            row["clean_point_membership"] = None
        candidates.append(row)
    valid = [row["index"] for row in candidates if row["passed"]]
    # Keep the best available positive component as failed diagnostic evidence.
    eligible = valid or list(cleaned) or list(range(len(values)))
    selected = max(eligible, key=lambda index: (scores[index], -index))
    if selected in cleaned:
        raw, clean, metrics = cleaned[selected]
    else:
        raw = values[selected] > 0
        clean = raw.copy()
        metrics = {
            "predicted_iou": float(scores[selected]),
            "raw_foreground_pixels": int(raw.sum()),
            "foreground_pixels": int(clean.sum()),
            "holes_filled_pixels": 0,
            "cleanup_unavailable": True,
        }
    return (
        raw,
        clean,
        {
            **metrics,
            "selected_index": selected,
            "candidate_count": len(values),
            "selection_passed": bool(valid),
            "candidates": candidates,
        },
    )


def contiguous_query_pool(hieradet):
    """Make Hiera pool a contiguous copy of its query (SAM 2 source left as it is).

    PyTorch 2.7 on Apple MPS returns wrong values from max_pool2d for the
    strided view of the query that `hieradet.do_pool` pools at every stage
    change (error about 5 on random data; 0 with a contiguous copy), so every
    block after the first stage was wrong on MPS. On the CPU the result is
    unchanged. Idempotent; returns the module."""
    original = hieradet.do_pool
    if getattr(original, "contiguous_query", False):
        return hieradet

    def do_pool(x, pool, norm=None):
        return original(x.contiguous() if pool is not None else x, pool, norm)

    do_pool.contiguous_query = True
    hieradet.do_pool = do_pool
    return hieradet


def _load_predictor(source, checkpoint, config, device):
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; choose CPU explicitly")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    sys.path.insert(0, str(source.resolve()))
    try:
        from sam2.build_sam import build_sam2
        from sam2.modeling.backbones import hieradet
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        contiguous_query_pool(hieradet)

        model = build_sam2(
            config, str(checkpoint), device=device, apply_postprocessing=False
        )
        predictor = SAM2ImagePredictor(model)
    finally:
        sys.path.pop(0)
    synchronize = torch.mps.synchronize if device == "mps" else lambda: None
    synchronize()
    return (
        predictor,
        torch.inference_mode,
        synchronize,
        {"torch": torch.__version__, "device": device, "hiera_query_pool": "contiguous copy (MPS max_pool2d fix)"},
    )


def run(
    images,
    coarse_masks,
    output,
    *,
    source,
    checkpoint,
    device="cpu",
    views=0,
    model_config=MODEL_CONFIG,
    prompts_json=None,
    multimask=False,
    automatic_cues=False,
    board_cameras=None,
    board_calibration=None,
    board_is_background=False,
    preserve_holes=False,
):
    if not isinstance(preserve_holes, bool):
        raise ValueError("preserve_holes must be an explicit boolean")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "schema": "photo_only_sam21_segmentation_v1",
        "status": "running",
        "experimental": True,
        "quality_accepted": False,
        "reference_used": False,
        "geometry_read": False,
        "configuration": {
            "images": str(images.resolve()),
            "coarse_masks": str(coarse_masks.resolve()),
            "source": str(source.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "device": device,
            "views": views,
            "model_config": model_config,
            "torch_threads": 2,
            "apply_postprocessing": False,
            "multimask_output": multimask,
            "automatic_cues": automatic_cues,
            "preserve_holes": preserve_holes,
            "prompts_json": str(prompts_json.resolve())
            if prompts_json is not None
            else None,
        },
        "prompt_recipe": "Frozen union of selected coarse-mask bounds without dilation; positive point is each coarse mask's maximum inscribed-distance point (row-major tie).",
        "mask_role": "inferred semantic object silhouette; accuracy unverified",
        "cleanup": (
            "Keep highest-scored prediction containing the positive prompt, retain its prompt-connected component, preserve background holes."
            if preserve_holes
            else "Keep highest-scored prediction containing the positive prompt, retain its prompt-connected component, fill holes. Real object holes may be filled."
        ),
        "implementation_sha256": digest(Path(__file__)),
        "rows": [],
    }
    paths = None
    board_context = None
    error = None
    try:
        if not isinstance(board_is_background, bool):
            raise ValueError("board_is_background must be an explicit boolean")
        board_feedback_enabled = (
            any(value is not None for value in (board_cameras, board_calibration))
            or board_is_background
        )
        if board_feedback_enabled and (
            board_cameras is None
            or board_calibration is None
            or not board_is_background
        ):
            raise ValueError(
                "board cameras, calibration, and board-is-background are required together"
            )
        if board_feedback_enabled and (
            prompts_json is not None or automatic_cues or not multimask
        ):
            raise ValueError(
                "board feedback requires automatic baseline prompts, no automatic-cues or overrides, and explicit multimask=True"
            )
        if device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        if not isinstance(multimask, bool):
            raise ValueError("multimask must be an explicit boolean")
        if not isinstance(automatic_cues, bool):
            raise ValueError("automatic_cues must be an explicit boolean")
        if Path(model_config).is_absolute() or ".." in Path(model_config).parts:
            raise ValueError("model config must be relative to source/sam2")
        paths = selected_images(images, views)
        report["source_inputs_before"] = input_inventory(paths, coarse_masks)
        report["model_before"] = model_inventory(source, checkpoint, model_config)
        if board_feedback_enabled:
            from scripts.turntable_mesh import board_feedback

            board_context = board_feedback.prepare_context(
                images,
                coarse_masks,
                board_cameras,
                board_calibration,
                [path.name for path in paths],
                board_is_background=True,
            )
            report["geometry_read"] = True
            report["board_feedback_prior"] = (
                "The photo-derived checkerboard is background; verified visible corners inside the baseline silhouette become negative cues. Cached photo-board poses are reused, without new pose recovery or reference geometry."
            )
            report["configuration"].update(
                board_cameras=str(board_cameras.resolve()),
                board_calibration=str(board_calibration.resolve()),
                board_is_background=True,
            )
            report["board_context_before"] = board_context["source_hashes_before"]
            report["runtime_counts"] = {"model_load": 1, "set_image": 0, "predict": 0}
            report["feedback_prompts"] = []
        prompts, dimensions = frozen_prompts(
            paths, coarse_masks, automatic_cues=automatic_cues
        )
        if automatic_cues:
            report["automatic_cue_prior"] = (
                "Object lies inside the common coarse-support bounding box expanded by 20% of its largest extent, minimum 24 pixels. Negatives are strictly outside this expanded box; out-of-image cues are skipped. Additional positives maximize inscribed distance within each quadrant, with clearance at least max(2px, 25% of global maximum) and separation from the primary point at least max(2px, 20% of the smaller support-box extent). This prior may be wrong for incomplete coarse masks; adjacent background inside the box can remain. Explicit photo overrides replace these generated points for named views."
            )
        if prompts_json is not None:
            report["prompt_overrides_sha256_before"] = digest(prompts_json)
            prompts = apply_prompt_overrides(
                prompts, prompts_json, report["source_inputs_before"], dimensions
            )
            if digest(prompts_json) != report["prompt_overrides_sha256_before"]:
                raise ValueError("photo prompt overrides changed during preparation")
        report["prompts"] = prompts
        if input_inventory(paths, coarse_masks) != report["source_inputs_before"]:
            raise ValueError(
                "source photographs or coarse masks changed during prompt preparation"
            )
        (output / "prompts.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        tick = time.monotonic()
        predictor, inference_mode, synchronize, versions = _load_predictor(
            source, checkpoint, model_config, device
        )
        report["model_load_seconds"] = time.monotonic() - tick
        report["runtime"] = versions
        for directory in ("masks", "raw-masks"):
            (output / directory).mkdir()
        if board_context is not None:
            for directory in ("baseline-masks", "baseline-raw-masks"):
                (output / directory).mkdir()
        for path, prompt in zip(paths, prompts):
            tick = time.monotonic()
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            with inference_mode():
                predictor.set_image(rgb)
                if board_context is not None:
                    synchronize()
                    embedding_seconds = time.monotonic() - tick
                    baseline_prediction_tick = time.monotonic()
                    report["runtime_counts"]["set_image"] += 1
                    report["runtime_counts"]["predict"] += 1
                masks, scores, _ = predictor.predict(
                    box=np.asarray(prompt["box_xyxy"], np.float32),
                    point_coords=np.asarray(
                        prompt.get("points_xy", [prompt["point_xy"]]), np.float32
                    ),
                    point_labels=np.asarray(prompt.get("point_labels", [1]), np.int32),
                    multimask_output=multimask,
                )
            synchronize()
            if board_context is not None:
                baseline_prediction_seconds = (
                    time.monotonic() - baseline_prediction_tick
                )
            if multimask:
                raw, clean, metrics = select_prediction(
                    masks, scores, prompt, dimensions, preserve_holes=preserve_holes
                )
            else:
                raw, clean, metrics = clean_prediction(
                    masks,
                    scores,
                    prompt["point_xy"],
                    dimensions,
                    preserve_holes=preserve_holes,
                )
            for directory, mask in (("masks", clean), ("raw-masks", raw)):
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    output / directory / (path.name + ".png")
                )
            row = {
                "name": path.name,
                "status": "candidate",
                "seconds": time.monotonic() - tick,
                **metrics,
                "raw_point_membership": point_membership(raw, prompt),
                "clean_point_membership": point_membership(clean, prompt),
                "mask_sha256": digest(output / "masks" / (path.name + ".png")),
                "raw_mask_sha256": digest(output / "raw-masks" / (path.name + ".png")),
            }
            report["rows"].append(row)
            if board_context is not None:
                row.update(
                    embedding_seconds=embedding_seconds,
                    baseline_prediction_seconds=baseline_prediction_seconds,
                )
                for directory, mask in (
                    ("baseline-masks", clean),
                    ("baseline-raw-masks", raw),
                ):
                    Image.fromarray(mask.astype(np.uint8) * 255).save(
                        output / directory / (path.name + ".png")
                    )
                row["baseline"] = dict(row)
            try:
                if multimask and not metrics["selection_passed"]:
                    raise ValueError(
                        "no SAM mask candidate satisfies all photographic point cues after cleanup"
                    )
                verify_cleaned_points(clean, prompt)
            except ValueError as failure:
                row.update(status="failed", error=str(failure))
                raise
            row["status"] = "complete_unreviewed"
            if board_context is not None:
                row["baseline"]["status"] = "complete_unreviewed"
                with Image.open(coarse_masks / (path.name + ".png")) as image:
                    coarse = np.array(image) > 0
                camera = board_context["by_name"][path.name]
                cues_tick = time.monotonic()
                chosen, checks = board_feedback.verified_corners(
                    rgb,
                    coarse,
                    np.asarray(camera["rotation"], float),
                    np.asarray(camera["translation"], float),
                    board_context["k"],
                    board_context["distortion"],
                    board_context["pattern"],
                    board_context["square_size"],
                )
                negatives = [point for point in chosen if clean[point[1], point[0]]]
                final_prompt = dict(prompt)
                final_prompt["points_xy"] = (
                    list(prompt.get("points_xy", [prompt["point_xy"]])) + negatives
                )
                final_prompt["point_labels"] = list(prompt.get("point_labels", [1])) + [
                    0
                ] * len(negatives)
                row["board_feedback"] = {
                    "seconds": time.monotonic() - cues_tick,
                    "chosen_corners": chosen,
                    "checks": checks,
                    "retained_negatives": negatives,
                    "feedback_applied": bool(negatives),
                    "final_prompt": final_prompt,
                    "method": "verified raw-RGB checkerboard contrast; coarse exclusion; eight farthest corners; baseline foreground inclusion",
                    "source": "bound cached photo-derived board cameras and raw lens calibration",
                }
                report["feedback_prompts"].append(
                    {
                        "name": path.name,
                        "rgb_sha256": report["source_inputs_before"][
                            len(report["rows"]) - 1
                        ]["rgb_sha256"],
                        "points_xy": final_prompt["points_xy"],
                        "point_labels": final_prompt["point_labels"],
                    }
                )
                if negatives:
                    feedback_tick = time.monotonic()
                    with inference_mode():
                        report["runtime_counts"]["predict"] += 1
                        masks, scores, _ = predictor.predict(
                            box=np.asarray(final_prompt["box_xyxy"], np.float32),
                            point_coords=np.asarray(
                                final_prompt["points_xy"], np.float32
                            ),
                            point_labels=np.asarray(
                                final_prompt["point_labels"], np.int32
                            ),
                            multimask_output=multimask,
                        )
                    synchronize()
                    if multimask:
                        raw, clean, metrics = select_prediction(
                            masks,
                            scores,
                            final_prompt,
                            dimensions,
                            preserve_holes=preserve_holes,
                        )
                    else:
                        raw, clean, metrics = clean_prediction(
                            masks,
                            scores,
                            final_prompt["point_xy"],
                            dimensions,
                            preserve_holes=preserve_holes,
                        )
                    for directory, mask in (("masks", clean), ("raw-masks", raw)):
                        Image.fromarray(mask.astype(np.uint8) * 255).save(
                            output / directory / (path.name + ".png")
                        )
                    row.update(
                        **metrics,
                        status="candidate",
                        feedback_seconds=time.monotonic() - feedback_tick,
                        raw_point_membership=point_membership(raw, final_prompt),
                        clean_point_membership=point_membership(clean, final_prompt),
                        mask_sha256=digest(output / "masks" / (path.name + ".png")),
                        raw_mask_sha256=digest(
                            output / "raw-masks" / (path.name + ".png")
                        ),
                    )
                    try:
                        if multimask and not metrics["selection_passed"]:
                            raise ValueError(
                                "no SAM feedback candidate satisfies all photographic point cues after cleanup"
                            )
                        verify_cleaned_points(clean, final_prompt)
                    except ValueError as failure:
                        row.update(status="failed", error=str(failure))
                        raise
                    row["status"] = "complete_unreviewed"
                row["seconds"] = time.monotonic() - tick
        if board_context is not None:
            (output / "feedback-prompts.json").write_text(
                json.dumps(
                    {
                        "schema": "photo_only_segmentation_prompts_v1",
                        "images": report["feedback_prompts"],
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
        report["status"] = "complete_unreviewed"
    except Exception as failure:
        error = failure
        report.update(
            status="failed",
            error={"type": type(failure).__name__, "message": str(failure)},
        )
    finally:
        try:
            if board_context is not None:
                report["board_context_after"] = {
                    key: digest(path)
                    for key, path in board_context["sealed_paths"].items()
                }
            if paths is not None and "source_inputs_before" in report:
                report["source_inputs_after"] = input_inventory(paths, coarse_masks)
                report["model_after"] = model_inventory(
                    source, checkpoint, model_config
                )
                if board_context is not None:
                    board_feedback.verify_context(board_context)
                if (
                    prompts_json is not None
                    and "prompt_overrides_sha256_before" in report
                ):
                    report["prompt_overrides_sha256_after"] = digest(prompts_json)
                    if (
                        report["prompt_overrides_sha256_after"]
                        != report["prompt_overrides_sha256_before"]
                    ):
                        raise ValueError(
                            "photo prompt overrides changed during segmentation"
                        )
                if (
                    report["source_inputs_after"] != report["source_inputs_before"]
                    or report["model_after"] != report["model_before"]
                    or digest(Path(__file__)) != report["implementation_sha256"]
                ):
                    raise ValueError(
                        "source photographs, coarse masks, model, or implementation changed during segmentation"
                    )
        except Exception as failure:
            report["provenance_error"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
            report["status"] = "failed"
            error = error or failure
        report["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if error is not None:
        raise error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "coarse-masks", "output", "source", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--views", type=int, default=0)
    parser.add_argument("--model-config", default=MODEL_CONFIG)
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="explicitly request multiple SAM hypotheses; select only candidates passing all photo point cues after cleanup",
    )
    parser.add_argument(
        "--preserve-holes",
        action="store_true",
        help="retain background holes in the prompt-connected mask, such as a visible handle opening",
    )
    parser.add_argument(
        "--automatic-cues",
        action="store_true",
        help="optional coarse-interior positives and distant negatives; explicitly assumes object containment within common coarse box expanded by 20 percent (at least 24 pixels)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="optional photo-hash-bound explicit positive/negative point corrections JSON",
    )
    parser.add_argument(
        "--board-cameras",
        type=Path,
        help="optional bound photo-board cameras for same-embedding background feedback",
    )
    parser.add_argument("--board-calibration", type=Path)
    parser.add_argument(
        "--board-is-background",
        action="store_true",
        help="explicit photo-derived board background prior; requires board cameras and calibration",
    )
    args = parser.parse_args()
    report = run(
        args.images,
        args.coarse_masks,
        args.output,
        source=args.source,
        checkpoint=args.checkpoint,
        device=args.device,
        views=args.views,
        model_config=args.model_config,
        prompts_json=args.prompts,
        multimask=args.multimask,
        automatic_cues=args.automatic_cues,
        preserve_holes=args.preserve_holes,
        board_cameras=args.board_cameras,
        board_calibration=args.board_calibration,
        board_is_background=args.board_is_background,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "masks": str((args.output / "masks").resolve()),
                "seconds": report["seconds"],
                "quality_accepted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
