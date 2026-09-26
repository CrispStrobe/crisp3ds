#!/usr/bin/env python3
"""Read-only expansion of the frozen held-out closed three-view cycles."""

import collections
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil

import scripts.sparse_verify.verify as verifier_math
from scripts.sparse_verify.verify import (
    cameras, closed_cycles, dot, images, mt, mv, pair_sampson,
    project, sha256, stats, undistort,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"build-opencv/colmap-sparse/heldout-v2-001"
OUTPUT = ROOT/"build-opencv/three-view-audit/run-001"
MIN_PARALLAX_DEG = 1.0
CROP_SIZE = 192


def closest_ray_midpoint(center_a, direction_a, center_b, direction_b):
    """Return closest-ray midpoint, gap, and forward ray parameters."""
    cosine = max(-1.0, min(1.0, dot(direction_a, direction_b)))
    denominator = 1-cosine*cosine
    if denominator < 1e-12:
        raise ValueError("parallel rays")
    delta = [v-u for u, v in zip(center_a, center_b)]
    right_a, right_b = dot(direction_a, delta), -dot(direction_b, delta)
    sa = (right_a+cosine*right_b)/denominator
    sb = (cosine*right_a+right_b)/denominator
    pa = [v+sa*d for v, d in zip(center_a, direction_a)]
    pb = [v+sb*d for v, d in zip(center_b, direction_b)]
    return ([(u+v)/2 for u, v in zip(pa, pb)],
            math.sqrt(sum((u-v)**2 for u, v in zip(pa, pb))), sa, sb)


def unit_world_ray(image, camera, xy):
    normalized = list(undistort(camera, xy))+[1.0]
    direction = mv(mt(image["R"]), normalized)
    length = math.sqrt(dot(direction, direction))
    return [value/length for value in direction]


def longest_baseline_index(baseline_pairs):
    available = [(item["camera_center_distance"], tuple(item["images"]), i)
                 for i, item in enumerate(baseline_pairs)
                 if item["camera_center_distance"] is not None]
    return sorted(available, key=lambda item: (-item[0], item[1]))[0][2] if available else None


def orient(cycle, first, second, third, image_by_name, cameras_by_id, features):
    a, b, c = cycle[first], cycle[second], cycle[third]
    result = dict(source=[a[0], a[1]], second=[b[0], b[1]], target=[c[0], c[1]],
                  status="unregistered", ray_parallax_deg=None,
                  closest_ray_gap=None, first_pair_reprojection_px=None,
                  second_pair_reprojection_px=None, third_reprojection_px=None,
                  predicted_third_xy=None, depths=None, point_xyz=None)
    if any(name not in image_by_name for name, _ in (a, b, c)):
        return result
    ia, ib, ic = [image_by_name[name] for name, _ in (a, b, c)]
    ca, cb, cc = [cameras_by_id[v["camera_id"]] for v in (ia, ib, ic)]
    try:
        da = unit_world_ray(ia, ca, features[a[0]][a[1]])
        db = unit_world_ray(ib, cb, features[b[0]][b[1]])
    except (ValueError, OverflowError, ZeroDivisionError):
        result["status"] = "invalid_distortion"
        return result
    cosine = max(-1.0, min(1.0, dot(da, db)))
    parallax = math.degrees(math.acos(cosine))
    result["ray_parallax_deg"] = parallax
    if parallax < MIN_PARALLAX_DEG:
        result["status"] = "low_parallax"
        return result
    try:
        point, gap, _, _ = closest_ray_midpoint(ia["center"], da, ib["center"], db)
    except ValueError:
        result["status"] = "invalid_triangulation"
        return result
    result["point_xyz"] = point
    result["closest_ray_gap"] = gap
    depths = []
    camera_points = []
    for image in (ia, ib, ic):
        xyz = [u+v for u, v in zip(mv(image["R"], point), image["t"])]
        depths.append(xyz[2])
        camera_points.append(xyz)
    result["depths"] = depths
    if min(depths) <= 0:
        result["status"] = "nonpositive_depth"
        return result
    for key, camera, camera_xyz, ref in (
            ("first_pair_reprojection_px", ca, camera_points[0], a),
            ("second_pair_reprojection_px", cb, camera_points[1], b),
            ("third_reprojection_px", cc, camera_points[2], c)):
        predicted = project(camera, camera_xyz)
        observed = features[ref[0]][ref[1]]
        result[key] = math.hypot(predicted[0]-observed[0], predicted[1]-observed[1])
        if key == "third_reprojection_px":
            result["predicted_third_xy"] = list(predicted)
    result["status"] = "scored"
    return result


def describe_cycles(manifest, model_dir):
    cameras_by_id = cameras(model_dir/"cameras.txt")
    ims = images(model_dir/"images.txt")
    by_name = {im["name"]: im for im in ims.values()}
    features = {im["name"]: im["features"] for im in manifest["images"]}
    pairs = [((m["image1"], m["feature1"]), (m["image2"], m["feature2"]))
             for m in manifest["heldout_pairs"]]
    cycles = closed_cycles(pairs)
    output = []
    for index, cycle in enumerate(cycles):
        refs = [dict(image=name, feature_id=fid, xy=list(features[name][fid][:2]))
                for name, fid in cycle]
        edges = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = cycle[i], cycle[j]
            distance = None
            if a[0] in by_name and b[0] in by_name:
                try:
                    distance = pair_sampson(by_name[a[0]], by_name[b[0]],
                                            cameras_by_id[by_name[a[0]]["camera_id"]],
                                            cameras_by_id[by_name[b[0]]["camera_id"]],
                                            features[a[0]][a[1]], features[b[0]][b[1]])
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass
            edges.append(dict(image1=a[0], feature1=a[1], image2=b[0], feature2=b[1],
                              sqrt_sampson_px=distance))
        orientations = [orient(cycle, *indices, by_name, cameras_by_id, features)
                        for indices in ((0, 1, 2), (0, 2, 1), (1, 2, 0))]
        baseline_pairs = []
        for first, second, target in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            names = (cycle[first][0], cycle[second][0])
            length = None
            if all(name in by_name for name in names):
                centers = [by_name[name]["center"] for name in names]
                length = math.sqrt(sum((x-y)**2 for x, y in zip(*centers)))
            baseline_pairs.append(dict(images=list(names), target=cycle[target][0],
                                       camera_center_distance=length))
        chosen = longest_baseline_index(baseline_pairs)
        output.append(dict(id=index, refs=refs, edges=edges,
                           lexical=orientations[0], all_orientations=orientations,
                           camera_baselines=baseline_pairs,
                           longest_baseline_orientation_index=chosen,
                           longest_baseline=orientations[chosen] if chosen is not None else None))
    return output


def dependence(cycles):
    occurrence = collections.Counter((ref["image"], ref["feature_id"])
                                     for cycle in cycles for ref in cycle["refs"])
    node_sets = [set((ref["image"], ref["feature_id"]) for ref in cycle["refs"])
                 for cycle in cycles]
    shared_pairs = sum(bool(a & b) for i, a in enumerate(node_sets)
                       for b in node_sets[i+1:])
    return dict(unique_feature_observations=len(occurrence),
                repeated_feature_observations=sum(v > 1 for v in occurrence.values()),
                maximum_cycles_per_feature=max(occurrence.values(), default=0),
                cycles_with_repeated_feature=sum(any(occurrence[node] > 1 for node in nodes)
                                                 for nodes in node_sets),
                cycle_pairs_sharing_feature=shared_pairs)


def graph_components(pairs, cycles):
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = collections.defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    ordered = sorted(groups.values(), key=lambda nodes: min(nodes))
    component_of = {node: i for i, nodes in enumerate(ordered) for node in nodes}
    component_report = []
    for i, nodes in enumerate(ordered):
        by_image = collections.Counter(name for name, _ in nodes)
        component_report.append(dict(id=i, feature_count=len(nodes),
                                     image_count=len(by_image),
                                     same_image_feature_conflict=any(v > 1 for v in by_image.values()),
                                     max_features_one_image=max(by_image.values())))
    failed = collections.Counter()
    for cycle in cycles:
        node = (cycle["refs"][0]["image"], cycle["refs"][0]["feature_id"])
        cid = component_of[node]
        cycle["graph_component_id"] = cid
        component_report[cid].setdefault("cycle_ids", []).append(cycle["id"])
        if (cycle["lexical"]["status"] == "scored" and
                cycle["lexical"]["third_reprojection_px"] > 4):
            failed[cid] += 1
    for item in component_report:
        item["frozen_failures_above_4px"] = failed[item["id"]]
    return dict(component_count=len(ordered),
                components_with_same_image_feature_conflict=sum(
                    item["same_image_feature_conflict"] for item in component_report),
                failed_cycles_in_conflicted_components=sum(
                    item["frozen_failures_above_4px"] for item in component_report
                    if item["same_image_feature_conflict"]),
                components_with_frozen_failures=sum(bool(v) for v in failed.values()),
                largest_frozen_failure_cluster=max(failed.values(), default=0),
                components=component_report)


def summary(cycles):
    frozen = [c["lexical"] for c in cycles]
    orientations = [o for c in cycles for o in c["all_orientations"]]
    counts = collections.Counter(o["status"] for o in frozen)
    all_counts = collections.Counter(o["status"] for o in orientations)
    fixed = [c for c in cycles if c["lexical"]["status"] == "scored"]
    policy = [c["longest_baseline"] for c in fixed]
    policy_values = [o["third_reprojection_px"] if o and o["status"] == "scored" else None
                     for o in policy]
    finite_policy = [v for v in policy_values if v is not None]
    all_policy = [c["longest_baseline"] for c in cycles]
    return dict(closed_cycles=len(cycles), registered_cycles=len(cycles)-counts["unregistered"],
                lexical_status_counts=dict(counts),
                lexical_third_reprojection=stats([o["third_reprojection_px"]
                                                  for o in frozen if o["status"] not in
                                                  ("unregistered", "low_parallax", "invalid_triangulation", "invalid_distortion")]),
                all_orientation_status_counts=dict(all_counts),
                all_orientation_third_reprojection=stats([o["third_reprojection_px"]
                                                          for o in orientations if o["status"] == "scored"]),
                longest_baseline_all_cycles_status_counts=dict(collections.Counter(
                    o["status"] if o is not None else "unregistered" for o in all_policy)),
                longest_baseline_on_fixed_lexical_eligible=dict(
                    original_eligible_count=len(fixed),
                    target_changed_count=sum(c["longest_baseline_orientation_index"] != 0 for c in fixed),
                    status_counts=dict(collections.Counter(o["status"] for o in policy)),
                    finite_reprojection=stats(finite_policy),
                    within_4px_count=sum(v <= 4 for v in finite_policy),
                    missing_inclusive_bad_over_4px_count=sum(v is None or v > 4 for v in policy_values)),
                dependence=dependence(cycles))


def verify_frozen_counts(result, saved):
    expected = saved["heldout"]["three_view"]
    if result["closed_cycles"] != expected["closed_cycles"] or result["registered_cycles"] != expected["registered_cycles"]:
        raise ValueError("closed or registered cycle count differs from frozen verifier")
    for status, key in (("low_parallax", "low_parallax"),
                        ("invalid_triangulation", "invalid_triangulation"),
                        ("invalid_distortion", "invalid_distortion"),
                        ("nonpositive_depth", "nonpositive_depth")):
        if result["lexical_status_counts"].get(status, 0) != expected[key]:
            raise ValueError(f"frozen {status} count mismatch")
    computed = result["lexical_third_reprojection"]
    original = expected["third_view_reprojection"]
    for key in ("count", "finite_count", "invalid_count", "within_1px", "within_2px", "within_4px"):
        if computed[key] != original[key]:
            raise ValueError(f"frozen third-view {key} mismatch")
    for key in ("median_px", "p90_px"):
        if not math.isclose(computed[key], original[key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"frozen third-view {key} mismatch")


def contact_html(cycles, manifest, output_dir):
    bad = [c for c in cycles if c["lexical"]["status"] == "scored"
           and c["lexical"]["third_reprojection_px"] > 4]
    good = sorted((c for c in cycles if c["lexical"]["status"] == "scored"),
                  key=lambda c: (c["lexical"]["third_reprojection_px"], c["id"]))[:10]
    image_paths = {im["name"]: im["path"] for im in manifest["images"]}
    def crop(ref, predicted=None):
        src = html.escape(os.path.relpath(image_paths[ref["image"]], output_dir), quote=True)
        x, y = ref["xy"]
        left = CROP_SIZE/2-x; top = CROP_SIZE/2-y
        label = html.escape(f'{ref["image"]} · feature {ref["feature_id"]} · ({x:.2f}, {y:.2f})')
        prediction = ""
        if predicted is not None:
            px, py = CROP_SIZE/2+predicted[0]-x, CROP_SIZE/2+predicted[1]-y
            if 0 <= px < CROP_SIZE and 0 <= py < CROP_SIZE:
                prediction = (f'<span class="pred-h" style="left:{px-11:.3f}px;top:{py-1:.3f}px"></span>'
                              f'<span class="pred-v" style="left:{px-1:.3f}px;top:{py-11:.3f}px"></span>')
            else:
                label += " · blue prediction outside crop"
        return (f'<figure><div class="crop"><img src="{src}" style="left:{left:.3f}px;top:{top:.3f}px" '
                f'width="768" height="512"><span class="cross-h"></span><span class="cross-v"></span>'
                f'{prediction}</div><figcaption>{label}</figcaption></figure>')
    sections = []
    for title, selected in (("Frozen errors above 4 px", bad), ("Ten lowest-error controls", good)):
        rows = []
        for c in selected:
            residual = c["lexical"]["third_reprojection_px"]
            parallax = c["lexical"]["ray_parallax_deg"]
            rows.append(f'<article id="cycle-{c["id"]}"><h3>Cycle {c["id"]}: third residual {residual:.3f} px; '
                        f'first-pair parallax {parallax:.3f}°</h3><div class="views">'
                        +"".join(crop(ref, c["lexical"]["predicted_third_xy"] if i == 2 else None)
                                 for i, ref in enumerate(c["refs"]))+"</div></article>")
        sections.append(f'<section><h2>{title} ({len(selected)})</h2>'+"".join(rows)+"</section>")
    index_rows = []
    selected_ids = {c["id"] for c in bad+good}
    for c in cycles:
        lexical = c["lexical"]
        label = f'{lexical["third_reprojection_px"]:.3f}' if lexical["third_reprojection_px"] is not None else "—"
        link = f'<a href="#cycle-{c["id"]}">{c["id"]}</a>' if c["id"] in selected_ids else str(c["id"])
        index_rows.append(f'<tr><td>{link}</td><td>{html.escape(lexical["status"])}</td><td>{label}</td></tr>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen three-view audit</title>'
            '<style>body{font:14px system-ui;margin:24px;color:#18212a;max-width:1250px}'
            'h1,h2,h3{margin:14px 0 8px}article{border-top:1px solid #bbb;padding:12px 0}'
            '.views{display:flex;gap:14px}.crop{width:192px;height:192px;overflow:hidden;position:relative;background:#ddd}'
            '.crop img{position:absolute;max-width:none}.cross-h,.cross-v,.pred-h,.pred-v{position:absolute;'
            'box-shadow:0 0 0 1px white;z-index:2}.cross-h,.cross-v{background:#e11}'
            '.pred-h,.pred-v{background:#06c}.cross-h{left:85px;top:95px;width:22px;height:2px}'
            '.cross-v{left:95px;top:85px;width:2px;height:22px}.pred-h{width:22px;height:2px}'
            '.pred-v{width:2px;height:22px}figure{margin:0;width:210px}'
            'figcaption{font-size:11px;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:2px 10px;'
            'border-bottom:1px solid #ddd;text-align:left}</style><h1>Frozen held-out three-view cycles</h1>'
            '<p>Crops use original local PNGs at native pixels. Red marks the held-out feature; blue '
            'marks the lexical prediction in the third view when it lies inside the crop. '
            'Cases are diagnostic; no semantic label is inferred.</p>'
            +"".join(sections)+'<section><h2>All 181 cycles</h2><p>Full per-cycle geometry is in report.json.</p>'
            '<table><tr><th>ID</th><th>Lexical status</th><th>Third error (px)</th></tr>'
            +"".join(index_rows)+'</table></section></html>')
    return page, len(bad), len(good)


def run(output_dir=OUTPUT, run_dir=RUN):
    output_dir = Path(output_dir); run_dir = Path(run_dir)
    if output_dir.exists():
        raise ValueError("output directory must be fresh")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < (10 << 30)+(50 << 20):
        raise ValueError("less than 10 GiB reserve plus 50 MiB output cap")
    inputs = [run_dir/"holdout.json", run_dir/"verification-supervisor.json"]
    inputs += [run_dir/"model_text"/name for name in ("cameras.txt", "images.txt", "points3D.txt")]
    manifest = json.loads(inputs[0].read_text())
    saved = json.loads(inputs[1].read_text())
    code_hashes_before = {"three_view_audit/run.py": sha256(__file__),
                          "sparse_verify/verify.py": sha256(verifier_math.__file__)}
    inputs += [Path(im["path"]) for im in manifest["images"]]
    hashes_before = {str(path): sha256(path) for path in inputs}
    if saved["integrity_status"] != "pass" or not saved["heldout_valid"]:
        raise ValueError("source held-out integrity audit is not valid")
    if code_hashes_before["sparse_verify/verify.py"] != saved["hashes"]["verifier"]:
        raise ValueError("imported geometry math differs from frozen verifier")
    if hashes_before[str(inputs[0])] != saved["hashes"]["manifest"]:
        raise ValueError("manifest differs from frozen independent verifier")
    for path in inputs[2:5]:
        if hashes_before[str(path)] != saved["hashes"]["model"][path.name]:
            raise ValueError(f"model differs from frozen independent verifier: {path.name}")
    for path in inputs[5:]:
        if hashes_before[str(path)] != saved["hashes"]["inputs"][str(path)]:
            raise ValueError(f"PNG differs from frozen independent verifier: {path}")
    cycles = describe_cycles(manifest, run_dir/"model_text")
    pairs = [((m["image1"], m["feature1"]), (m["image2"], m["feature2"]))
             for m in manifest["heldout_pairs"]]
    graph = graph_components(pairs, cycles)
    result = summary(cycles)
    result["raw_match_graph"] = graph
    verify_frozen_counts(result, saved)
    page, failures, controls = contact_html(cycles, manifest, output_dir)
    result.update(frozen_failures_above_4px=failures, low_error_controls=controls,
                  cycles=cycles, source_hashes_before=hashes_before,
                  source_hashes_after={str(path): sha256(path) for path in inputs},
                  code_hashes_before=code_hashes_before,
                  code_hashes_after={"three_view_audit/run.py": sha256(__file__),
                                     "sparse_verify/verify.py": sha256(verifier_math.__file__)})
    if result["source_hashes_before"] != result["source_hashes_after"]:
        raise ValueError("source changed during audit")
    if result["code_hashes_before"] != result["code_hashes_after"]:
        raise ValueError("code changed during audit")
    report_content = json.dumps(result, indent=2)+"\n"
    if len(report_content.encode())+len(page.encode()) > (50 << 20):
        raise ValueError("output exceeded 50 MiB")
    output_dir.mkdir()
    (output_dir/"report.json").write_text(report_content)
    (output_dir/"contacts.html").write_text(page)
    return {key: value for key, value in result.items()
            if key not in ("cycles", "source_hashes_before", "source_hashes_after")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(output_dir=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
