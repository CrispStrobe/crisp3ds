import { describe, expect, it } from "vitest";
import poseReady from "../../../tests/contracts/pose-ready-project.json";
import type { Project } from "./project";
import { parseSparseReport, validateSparseReport } from "./sparseReport";

const project = structuredClone(poseReady) as unknown as Project;
project.images = [
  { id: "view-001", path: "images/view-001.jpg", maskPath: "masks/view-001.png" },
  { id: "view-002", path: "images/view-002.jpg", maskPath: "masks/view-002.png" },
];
const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const report = {
  ok: true, sparseSchemaVersion: 1, projectId: project.id, units: "mm", coordinateFrame: "board",
  backend: { name: "OpenCV", version: "4.11.0", feature: "ORB", settings: { scaleLevels: 1, maxViews: 24, maxFeaturesPerView: 2000, pairWindow: 2, maxPairs: 45, maskErosionRadiusPx: 8, matchRatio: 0.75, minParallaxDeg: 1, maxReprojectionErrorPx: 2 } },
  views: project.images.map((image) => ({ imageId: image.id, path: image.path, maskPath: image.maskPath, featureCount: 23, rotation: identity, translationMm: [0, 0, 400] })),
  points: [{ id: 0, positionMm: [2, 3, 14], rmsReprojectionErrorPx: 0.3, maxReprojectionErrorPx: 0.5, minParallaxDeg: 2, observations: [{ imageId: "view-001", pixel: [120, 100] }, { imageId: "view-002", pixel: [125, 101] }] }],
  statistics: { candidateMatches: 4, acceptedTracks: 1, rejectedMatches: 3 },
};
const paths = (value: unknown, selected = project) => validateSparseReport(value, selected).map((issue) => issue.path);

describe("sparse report import", () => {
  it("accepts finite tracks tied to every project image and mask", () => {
    expect(paths(report)).toEqual([]);
    expect(parseSparseReport(JSON.stringify(report), project).report?.points[0].positionMm).toEqual([2, 3, 14]);
  });
  it("rejects foreign and stale image identities", () => {
    expect(paths({ ...report, projectId: "other" })).toContain("projectId");
    expect(paths({ ...report, views: [{ ...report.views[0], maskPath: "masks/other.png" }, report.views[1]] })).toContain("views[0].imageId");
    const changed = structuredClone(project);
    changed.images[0].path = "images/replaced.jpg";
    expect(paths(report, changed)).toContain("views[0].imageId");
  });
  it("rejects malformed geometry and impossible residuals", () => {
    expect(parseSparseReport("{", project).report).toBeUndefined();
    expect(paths({ ...report, points: [{ ...report.points[0], positionMm: [NaN, 0, 0] }] })).toContain("points[0].positionMm");
    expect(paths({ ...report, points: [{ ...report.points[0], positionMm: [1e7, 0, 0] }] })).toContain("points[0].positionMm");
    expect(paths({ ...report, views: [{ ...report.views[0], rotation: [-1, 0, 0, 0, 1, 0, 0, 0, 1] }, report.views[1]] })).toContain("views[0].rotation");
    expect(paths({ ...report, points: [{ ...report.points[0], maxReprojectionErrorPx: 2.1 }] })).toContain("points[0].residuals");
  });
  it("rejects duplicate IDs, missing or repeated observations, and invalid pixels", () => {
    expect(paths({ ...report, points: [report.points[0], report.points[0]], statistics: { ...report.statistics, acceptedTracks: 2 } })).toContain("points[1].id");
    expect(paths({ ...report, points: [{ ...report.points[0], observations: [report.points[0].observations[0]] }] })).toContain("points[0].observations");
    expect(paths({ ...report, points: [{ ...report.points[0], observations: [report.points[0].observations[0], report.points[0].observations[0]] }] })).toContain("points[0].observations[1]");
    expect(paths({ ...report, points: [{ ...report.points[0], observations: [{ ...report.points[0].observations[0], pixel: [-1, 0] }, report.points[0].observations[1]] }] })).toContain("points[0].observations[0].pixel");
  });
  it("checks finite numeric limits and proper rotations at the import boundary", () => {
    for (const position of [[Infinity, 0, 0], [0, -Infinity, 0], [0, 0, 1_000_001]]) {
      expect(paths({ ...report, points: [{ ...report.points[0], positionMm: position }] })).toContain("points[0].positionMm");
    }
    for (const rotation of [
      [1, 0, 0, 0, 1, 0, 0, 0, -1], // reflection
      [1, 0.1, 0, 0, 1, 0, 0, 0, 1], // shear
      [1, 0, 0, 0, 1, 0, 0, 0, Infinity],
    ]) {
      expect(paths({ ...report, views: [{ ...report.views[0], rotation }, report.views[1]] })).toContain("views[0].rotation");
    }
    expect(paths({ ...report, views: [{ ...report.views[0], translationMm: [0, NaN, 400] }, report.views[1]] })).toContain("views[0].translationMm");
    expect(paths({ ...report, backend: { ...report.backend, settings: { ...report.backend.settings, maxFeaturesPerView: NaN } } })).toContain("backend.settings");
    expect(paths({ ...report, points: [{ ...report.points[0], rmsReprojectionErrorPx: 0.6 }] })).toContain("points[0].residuals");
    expect(paths({ ...report, points: [{ ...report.points[0], minParallaxDeg: 0.9 }] })).toContain("points[0].minParallaxDeg");
    expect(paths({ ...report, statistics: { ...report.statistics, acceptedTracks: 2 } })).toContain("statistics.acceptedTracks");
  });
  it("checks observation identity and calibrated pixel bounds", () => {
    const [first, second] = report.points[0].observations;
    const width = project.calibration!.width, height = project.calibration!.height;
    const withPixel = (pixel: number[]) => ({ ...report, points: [{ ...report.points[0], observations: [{ ...first, pixel }, second] }] });
    expect(paths(withPixel([width - 0.001, height - 0.001]))).toEqual([]);
    for (const pixel of [[width, 0], [0, height], [NaN, 1], [1, Infinity], [-0.01, 1]]) {
      expect(paths(withPixel(pixel))).toContain("points[0].observations[0].pixel");
    }
    expect(paths({ ...report, points: [{ ...report.points[0], observations: [{ ...first, imageId: "foreign" }, second] }] })).toContain("points[0].observations[0]");
    expect(paths({ ...report, points: [{ ...report.points[0], observations: [first, { ...second, imageId: first.imageId }] }] })).toContain("points[0].observations[1]");
  });
});
