export const STAGES = ["calibration", "poses", "sparse", "dense", "mesh", "export"] as const;
export type Stage = typeof STAGES[number];
export type StageStatus = "pending" | "running" | "complete" | "failed" | "unavailable";

export interface ScanImage {
  id: string;
  path: string;
  maskPath?: string;
}

export interface Calibration {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  distortion: number[];
  distortionModel?: "opencv-radtan";
}

export interface BoardMarker {
  id: number;
  cornersMm: [number, number, number][];
}

export interface Board {
  type: "aruco" | "apriltag";
  markerSizeMm: number;
  rotatesWithObject: true;
  dictionary?: "DICT_4X4_50";
  markers?: BoardMarker[];
}

export interface Project {
  schemaVersion: 1;
  id: string;
  name: string;
  units: "mm";
  images: ScanImage[];
  calibration?: Calibration;
  board?: Board;
  stages: Partial<Record<Stage, StageStatus>>;
}

export interface ValidationIssue {
  path: string;
  message: string;
}

const statuses = new Set<StageStatus>(["pending", "running", "complete", "failed", "unavailable"]);
const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const nonempty = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const positive = (value: unknown): value is number => finite(value) && value > 0;

export function validRelativePath(path: unknown): path is string {
  if (!nonempty(path) || path !== path.trim() || path.includes("\\") || path.includes(":") || path.startsWith("/") || /[\u0000-\u001f]/.test(path)) return false;
  return path.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

export function validateProject(value: unknown): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const fail = (path: string, message: string) => issues.push({ path, message });
  if (!object(value)) return [{ path: "$", message: "Project must be a JSON object." }];
  if (value.schemaVersion !== 1) fail("schemaVersion", "Only schema version 1 is supported.");
  if (!nonempty(value.id)) fail("id", "Enter a project ID.");
  if (!nonempty(value.name)) fail("name", "Enter a project name.");
  if (value.units !== "mm") fail("units", "Units must be millimetres (mm).");

  if (!Array.isArray(value.images)) {
    fail("images", "Images must be an array.");
  } else {
    const ids = new Set<string>();
    value.images.forEach((image, index) => {
      const base = `images[${index}]`;
      if (!object(image)) return fail(base, "Image must be an object.");
      if (!nonempty(image.id)) fail(`${base}.id`, "Enter an image ID.");
      else if (ids.has(image.id)) fail(`${base}.id`, "Image IDs must be unique.");
      else ids.add(image.id);
      if (!validRelativePath(image.path)) fail(`${base}.path`, "Use a relative path without empty, . or .. segments, colons, or backslashes.");
      if (image.maskPath !== undefined && !validRelativePath(image.maskPath)) fail(`${base}.maskPath`, "Use a relative mask path without traversal.");
    });
  }

  if (value.calibration !== undefined) {
    const c = value.calibration;
    if (!object(c)) fail("calibration", "Calibration must be an object.");
    else {
      for (const field of ["width", "height"] as const) if (!positive(c[field]) || !Number.isInteger(c[field])) fail(`calibration.${field}`, "Use a positive integer image dimension.");
      for (const field of ["fx", "fy"] as const) if (!positive(c[field])) fail(`calibration.${field}`, "Use a positive focal length.");
      for (const field of ["cx", "cy"] as const) if (!finite(c[field])) fail(`calibration.${field}`, "Use a finite principal point.");
      if (!Array.isArray(c.distortion) || !c.distortion.every(finite)) fail("calibration.distortion", "Distortion must be an array of finite numbers.");
      if (c.distortionModel !== undefined) {
        if (c.distortionModel !== "opencv-radtan") fail("calibration.distortionModel", "Only opencv-radtan is supported.");
        if (Array.isArray(c.distortion) && ![4, 5, 8].includes(c.distortion.length)) fail("calibration.distortion", "OpenCV radial-tangential distortion needs 4, 5 or 8 coefficients.");
      }
    }
  }

  if (value.board !== undefined) {
    const b = value.board;
    if (!object(b)) fail("board", "Board must be an object.");
    else {
      if (b.type !== "aruco" && b.type !== "apriltag") fail("board.type", "Choose ArUco or AprilTag.");
      if (!positive(b.markerSizeMm)) fail("board.markerSizeMm", "Marker size must be positive in mm.");
      if (b.rotatesWithObject !== true) fail("board.rotatesWithObject", "The marker board must rotate with the object.");
      if (b.dictionary !== undefined && b.dictionary !== "DICT_4X4_50") fail("board.dictionary", "Only DICT_4X4_50 is supported.");
      if (b.dictionary !== undefined && b.type !== "aruco") fail("board.dictionary", "A dictionary requires an ArUco board.");
      if (b.markers !== undefined) {
        if (b.type !== "aruco" || b.dictionary !== "DICT_4X4_50") fail("board.markers", "Markers require an ArUco DICT_4X4_50 board.");
        if (!Array.isArray(b.markers) || b.markers.length === 0) fail("board.markers", "Markers must be a nonempty array.");
        else {
          const markerIds = new Set<number>();
          b.markers.forEach((marker, index) => {
            const base = `board.markers[${index}]`;
            if (!object(marker)) return fail(base, "Marker must be an object.");
            if (!Number.isInteger(marker.id) || !finite(marker.id) || marker.id < 0 || marker.id > 49) fail(`${base}.id`, "Use a marker ID from 0 to 49.");
            else if (markerIds.has(marker.id)) fail(`${base}.id`, "Marker IDs must be unique.");
            else markerIds.add(marker.id);
            const corners = marker.cornersMm;
            if (!Array.isArray(corners) || corners.length !== 4 || !corners.every((corner) => Array.isArray(corner) && corner.length === 3 && corner.every(finite) && corner[2] === 0)) {
              fail(`${base}.cornersMm`, "Supply four finite [x, y, 0] corners in decoded TL, TR, BR, BL order.");
            } else if (positive(b.markerSizeMm) && !validSquare(corners as number[][], b.markerSizeMm)) {
              fail(`${base}.cornersMm`, "Corners must form a nondegenerate square of markerSizeMm in decoded TL, TR, BR, BL order.");
            }
          });
        }
      }
    }
  }

  if (!object(value.stages)) fail("stages", "Stages must be an object.");
  else for (const [stage, status] of Object.entries(value.stages)) {
    if (!STAGES.includes(stage as Stage)) fail(`stages.${stage}`, "Unknown stage name.");
    else if (!statuses.has(status as StageStatus)) fail(`stages.${stage}`, "Use pending, running, complete, failed or unavailable.");
  }
  return issues;
}

function validSquare(corners: number[][], size: number): boolean {
  const edge = (a: number, b: number) => [corners[b][0] - corners[a][0], corners[b][1] - corners[a][1]];
  const edges = [edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 0)];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1];
  const tolerance = Math.max(1e-8, size * 1e-4);
  if (!edges.every((v) => Math.abs(Math.hypot(v[0], v[1]) - size) <= tolerance)) return false;
  if (edges.some((v, i) => Math.abs(dot(v, edges[(i + 1) % 4])) > size * tolerance)) return false;
  return edges.every((v, i) => {
    const next = edges[(i + 1) % 4];
    return v[0] * next[1] - v[1] * next[0] > size * size * (1 - 1e-4);
  });
}

export function generateBoardGrid(columns: number, rows: number, markerSizeMm: number, gapMm: number, firstId = 0): BoardMarker[] {
  if (![columns, rows, firstId].every(Number.isInteger) || columns < 1 || rows < 1 || firstId < 0 || firstId + columns * rows > 50 || !positive(markerSizeMm) || !finite(gapMm) || gapMm < 0) throw new Error("Use 1–50 markers, IDs 0–49, positive size and nonnegative gap.");
  const pitch = markerSizeMm + gapMm;
  const width = columns * markerSizeMm + (columns - 1) * gapMm;
  const height = rows * markerSizeMm + (rows - 1) * gapMm;
  return Array.from({ length: columns * rows }, (_, index) => {
    const x = -width / 2 + (index % columns) * pitch;
    const y = -height / 2 + Math.floor(index / columns) * pitch;
    return { id: firstId + index, cornersMm: [[x, y, 0], [x + markerSizeMm, y, 0], [x + markerSizeMm, y + markerSizeMm, 0], [x, y + markerSizeMm, 0]] };
  });
}

export function newProject(name: string): Project {
  const title = name.trim() || "Untitled scan";
  const id = crypto.randomUUID();
  return {
    schemaVersion: 1,
    id,
    name: title,
    units: "mm",
    images: [],
    stages: Object.fromEntries(STAGES.map((stage) => [stage, "pending"])) as Project["stages"],
  };
}

export function parseProject(text: string): { project?: Project; issues: ValidationIssue[] } {
  let value: unknown;
  try { value = JSON.parse(text); }
  catch { return { issues: [{ path: "$", message: "This is not valid JSON." }] }; }
  const issues = validateProject(value);
  return issues.length ? { issues } : { project: value as Project, issues: [] };
}

export function readiness(project: Project): { label: string; details: string[] } {
  const details: string[] = [];
  if (!project.images.length) details.push("Add still-image paths to the project.");
  if (!project.calibration) details.push("Add camera calibration for the source image dimensions.");
  else if (project.calibration.distortionModel !== "opencv-radtan") details.push("Select the OpenCV radial-tangential distortion model for pose estimation.");
  if (!project.board) details.push("Describe the marker board that rotates with the object.");
  else {
    if (project.board.type !== "aruco" || project.board.dictionary !== "DICT_4X4_50") details.push("Select an ArUco DICT_4X4_50 board for pose estimation.");
    if (!project.board.markers?.length) details.push("Enter measured board marker IDs and corners.");
  }
  if (details.length) return { label: "Setup incomplete", details };
  const laterDetails = project.images.some((image) => !image.maskPath) ? ["Add object mask paths before sparse or dense reconstruction."] : [];
  return { label: "Pose inputs recorded", details: ["Pose estimation has not run. Image existence and pose quality are unverified.", ...laterDetails] };
}
