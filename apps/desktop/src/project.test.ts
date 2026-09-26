import { describe, expect, it } from "vitest";
import empty from "../../../tests/contracts/empty-project.json";
import calibrated from "../../../tests/contracts/calibrated-project.json";
import poseReady from "../../../tests/contracts/pose-ready-project.json";
import { generateBoardGrid, newProject, parseProject, readiness, validateProject, validRelativePath, type Project } from "./project";

describe("project contract v1", () => {
  it("accepts the shared empty and calibrated fixtures", () => {
    expect(validateProject(empty)).toEqual([]);
    expect(validateProject(calibrated)).toEqual([]);
    expect(validateProject(poseReady)).toEqual([]);
  });

  it("keeps a new empty scan valid but not ready", () => {
    const project = newProject("  A brick  ");
    expect(project.name).toBe("A brick");
    expect(validateProject(project)).toEqual([]);
    expect(readiness(project).label).toBe("Setup incomplete");
  });

  it.each(["/tmp/image.jpg", "../image.jpg", "images/../a.jpg", "images//a.jpg", "images/./a.jpg", "C:/a.jpg", "images\\a.jpg", "images/a.jpg/", "images/\u0001a.jpg"])("rejects unsafe path %s", (path) => {
    expect(validRelativePath(path)).toBe(false);
  });

  it("rejects duplicate image IDs and malformed calibration", () => {
    const project = structuredClone(calibrated);
    project.images.push({ ...project.images[0] });
    project.calibration.fx = -1;
    expect(validateProject(project).map((issue) => issue.path)).toEqual(expect.arrayContaining([`images[${project.images.length - 1}].id`, "calibration.fx"]));
  });

  it("rejects unknown stages and a stationary marker board", () => {
    const project = structuredClone(calibrated) as Record<string, unknown>;
    project.stages = { unfamiliar: "complete" };
    project.board = { type: "aruco", markerSizeMm: 20, rotatesWithObject: false };
    expect(validateProject(project).map((issue) => issue.path)).toEqual(expect.arrayContaining(["stages.unfamiliar", "board.rotatesWithObject"]));
  });

  it("does not replace a valid project with invalid JSON", () => {
    expect(parseProject("{").issues[0].path).toBe("$");
    expect(parseProject(JSON.stringify({ ...empty, schemaVersion: 2 })).project).toBeUndefined();
    expect(parseProject(JSON.stringify(empty)).project?.id).toBe(empty.id);
  });

  it("requires the additive pose metadata for setup readiness without changing legacy validity", () => {
    expect(readiness(calibrated as unknown as Project).label).toBe("Setup incomplete");
    expect(readiness(poseReady as unknown as Project).label).toBe("Pose inputs recorded");
    expect(poseReady.stages.poses).toBe("pending");
    const noMasks = structuredClone(poseReady);
    noMasks.images = noMasks.images.map(({ id, path }) => ({ id, path })) as typeof noMasks.images;
    expect(readiness(noMasks as unknown as Project).label).toBe("Pose inputs recorded");
    expect(readiness(noMasks as unknown as Project).details.join(" ")).toMatch(/mask paths before sparse/);
  });

  it("generates centered row-major marker geometry", () => {
    const markers = generateBoardGrid(2, 2, 20, 5, 0);
    expect(markers).toEqual(poseReady.board.markers);
    expect(validateProject({ ...poseReady, board: { ...poseReady.board, markers } })).toEqual([]);
    expect(() => generateBoardGrid(8, 7, 20, 5)).toThrow();
  });

  it("rejects unsupported distortion lengths and marker IDs", () => {
    const project = structuredClone(poseReady);
    project.calibration.distortion = [0, 0, 0, 0, 0, 0];
    project.board.markers[1].id = 0;
    project.board.markers[2].id = 50;
    expect(validateProject(project).map((issue) => issue.path)).toEqual(expect.arrayContaining(["calibration.distortion", "board.markers[1].id", "board.markers[2].id"]));
  });

  it("requires an ArUco dictionary and at least one marker when geometry is present", () => {
    const withoutDictionary = structuredClone(poseReady) as Record<string, any>;
    delete withoutDictionary.board.dictionary;
    expect(validateProject(withoutDictionary).map((issue) => issue.path)).toContain("board.markers");
    withoutDictionary.board.dictionary = "DICT_4X4_50";
    withoutDictionary.board.type = "apriltag";
    expect(validateProject(withoutDictionary).map((issue) => issue.path)).toEqual(expect.arrayContaining(["board.dictionary", "board.markers"]));
    withoutDictionary.board.type = "aruco";
    withoutDictionary.board.markers = [];
    expect(validateProject(withoutDictionary).map((issue) => issue.path)).toContain("board.markers");
  });

  it("rejects malformed and mirrored marker squares but accepts in-plane rotation", () => {
    const project = structuredClone(poseReady);
    project.board.markers[0].cornersMm[2][0] += 2;
    project.board.markers[1].cornersMm[0][2] = 1;
    project.board.markers[2].cornersMm.reverse();
    expect(validateProject(project).map((issue) => issue.path)).toEqual(expect.arrayContaining(["board.markers[0].cornersMm", "board.markers[1].cornersMm", "board.markers[2].cornersMm"]));
    const rotated = structuredClone(poseReady);
    const s = 20 / Math.sqrt(2);
    rotated.board.markers[0].cornersMm = [[0, -s, 0], [s, 0, 0], [0, s, 0], [-s, 0, 0]];
    expect(validateProject(rotated)).toEqual([]);
  });

  it("preserves extra marker metadata through project parsing", () => {
    const project = structuredClone(poseReady) as Record<string, any>;
    project.board.markers[0].printLabel = "top-left fixture";
    const parsed = parseProject(JSON.stringify(project));
    expect(parsed.issues).toEqual([]);
    expect((parsed.project?.board?.markers?.[0] as unknown as Record<string, unknown> | undefined)?.printLabel).toBe("top-left fixture");
  });
});
