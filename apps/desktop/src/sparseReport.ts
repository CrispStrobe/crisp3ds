import type { Project, ValidationIssue } from "./project";

export interface SparseView {
  imageId: string;
  path: string;
  maskPath: string;
  featureCount: number;
  rotation: number[];
  translationMm: number[];
}

export interface SparsePoint {
  id: number;
  positionMm: number[];
  rmsReprojectionErrorPx: number;
  maxReprojectionErrorPx: number;
  minParallaxDeg: number;
  observations: { imageId: string; pixel: number[] }[];
}

export interface SparseReport {
  ok: true;
  sparseSchemaVersion: 1;
  projectId: string;
  units: "mm";
  coordinateFrame: "board";
  backend: { name: "OpenCV"; version: string; feature: "ORB"; settings: Record<string, number | string | boolean> };
  views: SparseView[];
  points: SparsePoint[];
  statistics: { candidateMatches: number; acceptedTracks: number; rejectedMatches: number; [key: string]: number };
}

const MAX_POINTS = 100_000;
const MAX_OBSERVATIONS = 1_000_000;
const MAX_ABS_MM = 1_000_000;
const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => finite(value) && value >= 0;
const integer = (value: unknown): value is number => nonnegative(value) && Number.isSafeInteger(value);
const positiveText = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const vector = (value: unknown, length: number): value is number[] => Array.isArray(value) && value.length === length && value.every(finite);

function properRotation(matrix: number[]): boolean {
  const dot = (a: number, b: number) => matrix[a * 3] * matrix[b * 3] + matrix[a * 3 + 1] * matrix[b * 3 + 1] + matrix[a * 3 + 2] * matrix[b * 3 + 2];
  if ([0, 1, 2].some((row) => Math.abs(dot(row, row) - 1) > 1e-3)) return false;
  if ([[0, 1], [0, 2], [1, 2]].some(([a, b]) => Math.abs(dot(a, b)) > 1e-3)) return false;
  const determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  return Math.abs(determinant - 1) <= 1e-3;
}

export function validateSparseReport(value: unknown, project: Project): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const fail = (path: string, message: string) => issues.push({ path, message });
  if (!object(value)) return [{ path: "$", message: "Sparse report must be a JSON object." }];
  if (value.ok !== true) fail("ok", "Import a successful sparse report.");
  if (value.sparseSchemaVersion !== 1) fail("sparseSchemaVersion", "Only sparse report version 1 is supported.");
  if (value.projectId !== project.id) fail("projectId", "Report belongs to a different project.");
  if (value.units !== "mm") fail("units", "Point coordinates must use millimetres.");
  if (value.coordinateFrame !== "board") fail("coordinateFrame", "Points must use the board coordinate frame.");
  const settings = object(value.backend) && object(value.backend.settings) ? value.backend.settings : null;
  if (!object(value.backend) || value.backend.name !== "OpenCV" || !positiveText(value.backend.version) || value.backend.feature !== "ORB" || !settings) fail("backend", "Expected an OpenCV ORB backend with settings.");
  if (settings && Object.values(settings).some((item) => !(typeof item === "string" || typeof item === "boolean" || finite(item)))) fail("backend.settings", "Settings must contain finite numbers, strings, or booleans.");
  if (settings) {
    for (const key of ["scaleLevels", "maxViews", "maxFeaturesPerView", "pairWindow", "maxPairs"] as const) if (!integer(settings[key]) || settings[key] < 1) fail(`backend.settings.${key}`, "Setting must be a positive integer.");
    if (!integer(settings.maskErosionRadiusPx)) fail("backend.settings.maskErosionRadiusPx", "Mask erosion radius must be a nonnegative integer.");
    if (!finite(settings.matchRatio) || settings.matchRatio <= 0 || settings.matchRatio >= 1) fail("backend.settings.matchRatio", "Match ratio must be between 0 and 1.");
    if (!finite(settings.minParallaxDeg) || settings.minParallaxDeg <= 0 || settings.minParallaxDeg > 180) fail("backend.settings.minParallaxDeg", "Minimum parallax must be within 0–180 degrees.");
  }
  const maxResidual = settings?.maxReprojectionErrorPx;
  if (!finite(maxResidual) || maxResidual <= 0) fail("backend.settings.maxReprojectionErrorPx", "A positive per-observation pixel error limit is required.");

  const images = project.images;
  if (!Array.isArray(value.views) || value.views.length !== images.length || value.views.length < 2 || value.views.length > 64) fail("views", "Expected 2–64 views in project image order.");
  else value.views.forEach((view, index) => {
    const base = `views[${index}]`;
    const image = images[index];
    if (!object(view)) return fail(base, "View must be an object.");
    if (view.imageId !== image.id || view.path !== image.path || !image.maskPath || view.maskPath !== image.maskPath) fail(`${base}.imageId`, "View ID, image path, and mask path must match this project in order.");
    if (!integer(view.featureCount) || view.featureCount > 100_000) fail(`${base}.featureCount`, "Feature count must be a bounded nonnegative integer.");
    if (!vector(view.rotation, 9) || !properRotation(view.rotation)) fail(`${base}.rotation`, "Rotation must be a finite, proper row-major 3×3 matrix.");
    if (!vector(view.translationMm, 3) || view.translationMm.some((n: number) => Math.abs(n) > MAX_ABS_MM)) fail(`${base}.translationMm`, "Translation must be finite and within 1,000,000 mm.");
    if (settings && integer(settings.maxFeaturesPerView) && integer(view.featureCount) && view.featureCount > settings.maxFeaturesPerView) fail(`${base}.featureCount`, "Feature count exceeds the backend limit.");
  });
  if (settings && Array.isArray(value.views) && integer(settings.maxViews) && value.views.length > settings.maxViews) fail("views", "View count exceeds the backend limit.");

  if (!Array.isArray(value.points) || !value.points.length || value.points.length > MAX_POINTS) fail("points", `Expected 1–${MAX_POINTS.toLocaleString()} sparse points.`);
  else {
    const ids = new Set<number>();
    const imageIds = new Set(images.map((image) => image.id));
    const width = project.calibration?.width;
    const height = project.calibration?.height;
    let observationCount = 0;
    value.points.forEach((point, index) => {
      const base = `points[${index}]`;
      if (!object(point)) return fail(base, "Point must be an object.");
      if (!integer(point.id) || ids.has(point.id)) fail(`${base}.id`, "Point IDs must be unique nonnegative integers.");
      else ids.add(point.id);
      if (!vector(point.positionMm, 3) || point.positionMm.some((n: number) => Math.abs(n) > MAX_ABS_MM)) fail(`${base}.positionMm`, "Position must be finite and within 1,000,000 mm.");
      if (!nonnegative(point.rmsReprojectionErrorPx) || !nonnegative(point.maxReprojectionErrorPx) || (nonnegative(point.rmsReprojectionErrorPx) && nonnegative(point.maxReprojectionErrorPx) && point.rmsReprojectionErrorPx > point.maxReprojectionErrorPx + 1e-6) || (finite(maxResidual) && nonnegative(point.maxReprojectionErrorPx) && point.maxReprojectionErrorPx > maxResidual + 1e-6)) fail(`${base}.residuals`, "Residuals must be finite, ordered, and within the backend pixel error limit.");
      if (!finite(point.minParallaxDeg) || point.minParallaxDeg < 0 || point.minParallaxDeg > 180) fail(`${base}.minParallaxDeg`, "Parallax must be between 0 and 180 degrees.");
      else if (settings && finite(settings.minParallaxDeg) && point.minParallaxDeg + 1e-6 < settings.minParallaxDeg) fail(`${base}.minParallaxDeg`, "Track parallax is below the backend threshold.");
      if (!Array.isArray(point.observations) || point.observations.length < 2 || point.observations.length > images.length) fail(`${base}.observations`, "A track needs 2 or more project views, each used once.");
      else {
        observationCount += point.observations.length;
        const seen = new Set<string>();
        point.observations.forEach((observation, observationIndex) => {
          const path = `${base}.observations[${observationIndex}]`;
          if (!object(observation) || !positiveText(observation.imageId) || !imageIds.has(observation.imageId) || seen.has(observation.imageId)) return fail(path, "Observation must refer to a distinct project view.");
          seen.add(observation.imageId);
          if (!vector(observation.pixel, 2) || observation.pixel[0] < 0 || observation.pixel[1] < 0 || (finite(width) && observation.pixel[0] >= width) || (finite(height) && observation.pixel[1] >= height)) fail(`${path}.pixel`, "Pixel must be finite and inside the calibrated image.");
        });
      }
    });
    if (observationCount > MAX_OBSERVATIONS) fail("points", "Report exceeds the observation import limit.");
  }
  if (!object(value.statistics) || !integer(value.statistics.candidateMatches) || !integer(value.statistics.acceptedTracks) || !integer(value.statistics.rejectedMatches) || Object.values(value.statistics).some((n) => !integer(n))) fail("statistics", "Statistics must be nonnegative integer counters.");
  else if (Array.isArray(value.points) && value.statistics.acceptedTracks !== value.points.length) fail("statistics.acceptedTracks", "Accepted track count must equal the point count.");
  return issues;
}

export function parseSparseReport(source: string, project: Project): { report?: SparseReport; issues: ValidationIssue[] } {
  let value: unknown;
  try { value = JSON.parse(source); }
  catch { return { issues: [{ path: "$", message: "This is not valid JSON." }] }; }
  const issues = validateSparseReport(value, project);
  return issues.length ? { issues } : { report: value as unknown as SparseReport, issues: [] };
}
