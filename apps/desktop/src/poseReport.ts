import type { Project, ValidationIssue } from "./project";

export interface ImagePose {
  imageId: string;
  path: string;
  detectedMarkerIds: number[];
  ignoredMarkerIds: number[];
  cornerCount: number;
  rotation: number[];
  translationMm: number[];
  rmsReprojectionErrorPx: number;
  maxReprojectionErrorPx: number;
}

export interface PoseReport {
  ok: true;
  poseSchemaVersion: 1;
  projectId: string;
  units: "mm";
  convention: "object-to-camera";
  matrixStorage: "row-major";
  backend: {
    name: string;
    version: string;
    dictionary: "DICT_4X4_50";
    solver: string;
    maxRmsPx: number;
    maxErrorPx: number;
  };
  poses: ImagePose[];
}

const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => finite(value) && value >= 0;
const nonempty = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const vector = (value: unknown, size: number): value is number[] => Array.isArray(value) && value.length === size && value.every(finite);

function plausibleRotation(matrix: number[]): boolean {
  const dot = (rowA: number, rowB: number) => matrix[rowA * 3] * matrix[rowB * 3] + matrix[rowA * 3 + 1] * matrix[rowB * 3 + 1] + matrix[rowA * 3 + 2] * matrix[rowB * 3 + 2];
  if ([0, 1, 2].some((row) => Math.abs(dot(row, row) - 1) > 1e-3)) return false;
  if ([[0, 1], [0, 2], [1, 2]].some(([a, b]) => Math.abs(dot(a, b)) > 1e-3)) return false;
  const determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  return Math.abs(determinant - 1) <= 1e-3;
}

export function validatePoseReport(value: unknown, project: Project): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const fail = (path: string, message: string) => issues.push({ path, message });
  if (!record(value)) return [{ path: "$", message: "Pose report must be a JSON object." }];
  if (value.ok !== true) fail("ok", "Import a successful pose report.");
  if (value.poseSchemaVersion !== 1) fail("poseSchemaVersion", "Only pose report version 1 is supported.");
  if (value.projectId !== project.id) fail("projectId", "Report belongs to a different project.");
  if (value.units !== "mm") fail("units", "Pose translation must use millimetres.");
  if (value.convention !== "object-to-camera") fail("convention", "Pose convention must be object-to-camera.");
  if (value.matrixStorage !== "row-major") fail("matrixStorage", "Rotation storage must be row-major.");
  if (!record(value.backend)) fail("backend", "Backend description is missing.");
  else {
    if (value.backend.name !== "OpenCV" || !nonempty(value.backend.version) || value.backend.dictionary !== "DICT_4X4_50" || value.backend.solver !== "IPPE+LM") fail("backend", "Unsupported or incomplete pose backend description.");
    if (!nonnegative(value.backend.maxRmsPx) || !nonnegative(value.backend.maxErrorPx)) fail("backend", "Backend residual thresholds must be finite and nonnegative.");
  }
  if (!Array.isArray(value.poses) || value.poses.length !== project.images.length || value.poses.length === 0) {
    fail("poses", "Report needs one pose for every image in this project.");
  } else {
    const expected = new Map(project.images.map((image) => [image.id, image.path]));
    const seen = new Set<string>();
    const configured = new Set(project.board?.markers?.map((marker) => marker.id) ?? []);
    value.poses.forEach((pose, index) => {
      const base = `poses[${index}]`;
      if (!record(pose)) return fail(base, "Pose must be an object.");
      if (!nonempty(pose.imageId) || !expected.has(pose.imageId) || seen.has(pose.imageId) || expected.get(pose.imageId) !== pose.path) fail(`${base}.imageId`, "Pose image ID and path must match an unused project image.");
      else seen.add(pose.imageId);
      if (!Array.isArray(pose.detectedMarkerIds) || !pose.detectedMarkerIds.length || !pose.detectedMarkerIds.every((id) => Number.isInteger(id) && configured.has(id)) || new Set(pose.detectedMarkerIds).size !== pose.detectedMarkerIds.length) fail(`${base}.detectedMarkerIds`, "Detected IDs must be unique and belong to the configured board.");
      if (!Array.isArray(pose.ignoredMarkerIds) || !pose.ignoredMarkerIds.every((id) => Number.isInteger(id) && id >= 0 && id <= 49 && !configured.has(id)) || new Set(pose.ignoredMarkerIds).size !== pose.ignoredMarkerIds.length) fail(`${base}.ignoredMarkerIds`, "Ignored IDs must be distinct, unconfigured ArUco IDs 0–49.");
      if (!finite(pose.cornerCount) || !Number.isInteger(pose.cornerCount) || pose.cornerCount < 4 || !Array.isArray(pose.detectedMarkerIds) || pose.cornerCount !== pose.detectedMarkerIds.length * 4) fail(`${base}.cornerCount`, "Corner count must equal four per detected marker.");
      if (!vector(pose.rotation, 9) || !plausibleRotation(pose.rotation)) fail(`${base}.rotation`, "Rotation must be a finite, proper row-major 3×3 matrix.");
      if (!vector(pose.translationMm, 3)) fail(`${base}.translationMm`, "Translation must contain three finite millimetre values.");
      if (!nonnegative(pose.rmsReprojectionErrorPx) || !nonnegative(pose.maxReprojectionErrorPx) || (nonnegative(pose.rmsReprojectionErrorPx) && nonnegative(pose.maxReprojectionErrorPx) && pose.rmsReprojectionErrorPx > pose.maxReprojectionErrorPx + 1e-6)) fail(`${base}.residuals`, "Residuals must be finite, nonnegative, and max must be at least RMS.");
      if (record(value.backend) && nonnegative(value.backend.maxRmsPx) && nonnegative(value.backend.maxErrorPx) && ((nonnegative(pose.rmsReprojectionErrorPx) && pose.rmsReprojectionErrorPx > value.backend.maxRmsPx) || (nonnegative(pose.maxReprojectionErrorPx) && pose.maxReprojectionErrorPx > value.backend.maxErrorPx))) fail(`${base}.residuals`, "Successful report exceeds its backend residual thresholds.");
    });
  }
  return issues;
}

export function parsePoseReport(source: string, project: Project): { report?: PoseReport; issues: ValidationIssue[] } {
  let value: unknown;
  try { value = JSON.parse(source); }
  catch { return { issues: [{ path: "$", message: "This is not valid JSON." }] }; }
  const issues = validatePoseReport(value, project);
  return issues.length ? { issues } : { report: value as PoseReport, issues: [] };
}
