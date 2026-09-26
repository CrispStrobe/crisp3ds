import { describe, expect, it } from "vitest";
import poseReady from "../../../tests/contracts/pose-ready-project.json";
import type { Project } from "./project";
import { parsePoseReport, validatePoseReport } from "./poseReport";

const project = poseReady as unknown as Project;
const report = {
  ok: true,
  poseSchemaVersion: 1,
  projectId: project.id,
  units: "mm",
  convention: "object-to-camera",
  matrixStorage: "row-major",
  backend: { name: "OpenCV", version: "4.11.0", dictionary: "DICT_4X4_50", solver: "IPPE+LM", maxRmsPx: 3, maxErrorPx: 8 },
  poses: [{ imageId: "view-001", path: "images/view-001.jpg", detectedMarkerIds: [0, 1], ignoredMarkerIds: [8], cornerCount: 8, rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translationMm: [2, -3, 400], rmsReprojectionErrorPx: 0.2, maxReprojectionErrorPx: 0.4 }],
};

describe("pose report import", () => {
  it("accepts a complete report for the open project", () => {
    expect(validatePoseReport(report, project)).toEqual([]);
    expect(parsePoseReport(JSON.stringify(report), project).report?.poses[0].translationMm).toEqual([2, -3, 400]);
  });

  it("rejects malformed or foreign project reports", () => {
    expect(parsePoseReport("{", project).report).toBeUndefined();
    expect(validatePoseReport({ ...report, projectId: "another-scan" }, project).map((issue) => issue.path)).toContain("projectId");
    expect(validatePoseReport({ ...report, poses: [{ ...report.poses[0], path: "images/stale.jpg" }] }, project).map((issue) => issue.path)).toContain("poses[0].imageId");
  });

  it("rejects incomplete image sets, improper rotations, and invalid residuals", () => {
    expect(validatePoseReport({ ...report, poses: [] }, project).map((issue) => issue.path)).toContain("poses");
    const reflected = { ...report, poses: [{ ...report.poses[0], rotation: [-1, 0, 0, 0, 1, 0, 0, 0, 1] }] };
    expect(validatePoseReport(reflected, project).map((issue) => issue.path)).toContain("poses[0].rotation");
    const residual = { ...report, poses: [{ ...report.poses[0], maxReprojectionErrorPx: -1 }] };
    expect(validatePoseReport(residual, project).map((issue) => issue.path)).toContain("poses[0].residuals");
    const excessive = { ...report, poses: [{ ...report.poses[0], rmsReprojectionErrorPx: 3.1, maxReprojectionErrorPx: 8.1 }] };
    expect(validatePoseReport(excessive, project).map((issue) => issue.path)).toContain("poses[0].residuals");
    const duplicate = { ...report, poses: [{ ...report.poses[0], detectedMarkerIds: [0, 0], ignoredMarkerIds: [1] }] };
    expect(validatePoseReport(duplicate, project).map((issue) => issue.path)).toEqual(expect.arrayContaining(["poses[0].detectedMarkerIds", "poses[0].ignoredMarkerIds"]));
  });

  it("rejects a stale report after an image path changes", () => {
    const changed = structuredClone(project);
    changed.images[0].path = "images/replaced.jpg";
    expect(validatePoseReport(report, changed).map((issue) => issue.path)).toContain("poses[0].imageId");
  });
});
