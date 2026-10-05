import { describe, expect, it } from "vitest";
import {
  defaultFormValues,
  diffSettings,
  groupSettings,
  isChanged,
  parseFormValue,
  parseSettingsSchema,
  placeServerError,
  startRunBody,
  type SettingSpec,
} from "./settings";

/** A cut of what `GET /api/settings` returns. */
const PAYLOAD = {
  settings: [
    { name: "sizes", group: "Pyramid", meaning: "Longest canvas side per level", kind: "integer_list", default: [256, 512, 1024] },
    { name: "crop_padding", group: "Pyramid", meaning: "Pixels kept around each mask box", kind: "integer", default: 24 },
    { name: "neighbours", group: "Matching", meaning: "Neighbouring views", kind: "integer", default: 6 },
    { name: "best_of", group: "Matching", meaning: "Best scores averaged", kind: "integer", default: 3 },
    { name: "min_score", group: "Matching", meaning: "Lowest accepted score", kind: "number", default: 0.55 },
    { name: "aggregates", group: "Matching", meaning: "Cost blur per level", kind: "number_list", default: [1.0, 1.5, 2.0] },
    { name: "repair_masks", group: "Hull", meaning: "Mask repair on or off", kind: "boolean", default: true },
    { name: "grid", group: "Hull", meaning: "Voxels along the longest side", kind: "integer", default: 400 },
  ],
};

const specs = parseSettingsSchema(PAYLOAD);
const spec = (name: string): SettingSpec => specs.find((candidate) => candidate.name === name)!;

describe("settings schema", () => {
  it("groups in order of first appearance", () => {
    expect(groupSettings(specs).map((group) => [group.group, group.settings.length])).toEqual([
      ["Pyramid", 2],
      ["Matching", 4],
      ["Hull", 2],
    ]);
  });

  it("survives malformed rows and payloads", () => {
    expect(parseSettingsSchema(null)).toEqual([]);
    expect(parseSettingsSchema({ settings: "no" })).toEqual([]);
    const loose = parseSettingsSchema({ settings: [null, 5, { group: "x" }, { name: "odd", default: { a: 1 }, extra: true }] });
    expect(loose).toEqual([{ name: "odd", group: "Other", meaning: "", kind: "unknown", default: { a: 1 } }]);
  });

  it("writes defaults as form text, lists comma-separated", () => {
    expect(defaultFormValues(specs)).toMatchObject({
      sizes: "256, 512, 1024",
      crop_padding: "24",
      min_score: "0.55",
      aggregates: "1, 1.5, 2",
      repair_masks: true,
    });
  });
});

describe("parseFormValue", () => {
  it("reads typed values", () => {
    expect(parseFormValue(spec("grid"), " 96 ")).toEqual({ ok: true, value: 96 });
    expect(parseFormValue(spec("min_score"), "0.6")).toEqual({ ok: true, value: 0.6 });
    expect(parseFormValue(spec("min_score"), "1e-4")).toEqual({ ok: true, value: 0.0001 });
    expect(parseFormValue(spec("sizes"), "64,128")).toEqual({ ok: true, value: [64, 128] });
    expect(parseFormValue(spec("sizes"), " 64 , 128, ")).toEqual({ ok: true, value: [64, 128] });
    expect(parseFormValue(spec("sizes"), "64 128")).toEqual({ ok: true, value: [64, 128] });
    expect(parseFormValue(spec("aggregates"), "1, 1.5")).toEqual({ ok: true, value: [1, 1.5] });
    expect(parseFormValue(spec("repair_masks"), false)).toEqual({ ok: true, value: false });
  });

  it("explains what is wrong", () => {
    expect(parseFormValue(spec("grid"), "")).toMatchObject({ ok: false });
    expect(parseFormValue(spec("grid"), "96.5")).toEqual({ ok: false, error: "Needs a whole number." });
    expect(parseFormValue(spec("grid"), "many")).toMatchObject({ ok: false });
    expect(parseFormValue(spec("min_score"), "0,6")).toEqual({ ok: false, error: "Needs a number." });
    expect(parseFormValue(spec("sizes"), "")).toEqual({ ok: false, error: "Needs at least one value." });
    expect(parseFormValue(spec("sizes"), "64, big")).toMatchObject({ ok: false, error: expect.stringContaining('"big"') });
    expect(parseFormValue(spec("sizes"), "64, 128.5")).toMatchObject({ ok: false });
  });
});

describe("diffSettings", () => {
  it("sends nothing when nothing changed", () => {
    expect(diffSettings(specs, defaultFormValues(specs))).toEqual({ changed: {}, errors: {} });
  });

  it("sends only the values that differ from the defaults, typed", () => {
    const form = {
      ...defaultFormValues(specs),
      sizes: "64,128",
      grid: "96",
      aggregates: "1, 1",
      repair_masks: false,
      min_score: "0.550", // same number, written differently
      crop_padding: " 24", // same number, stray space
    };
    expect(diffSettings(specs, form)).toEqual({
      changed: { sizes: [64, 128], grid: 96, aggregates: [1, 1], repair_masks: false },
      errors: {},
    });
  });

  it("treats a list written differently but equal as unchanged, and a reordered one as changed", () => {
    expect(diffSettings(specs, { ...defaultFormValues(specs), sizes: "256,512,1024" }).changed).toEqual({});
    expect(diffSettings(specs, { ...defaultFormValues(specs), sizes: "512, 256, 1024" }).changed).toEqual({ sizes: [512, 256, 1024] });
  });

  it("goes back to nothing after a reset to defaults", () => {
    const edited = { ...defaultFormValues(specs), grid: "128" };
    expect(Object.keys(diffSettings(specs, edited).changed)).toEqual(["grid"]);
    expect(diffSettings(specs, defaultFormValues(specs)).changed).toEqual({});
  });

  it("reports invalid input per field and does not send it", () => {
    const result = diffSettings(specs, { ...defaultFormValues(specs), grid: "big", sizes: "", neighbours: "4" });
    expect(result.changed).toEqual({ neighbours: 4 });
    expect(Object.keys(result.errors).sort()).toEqual(["grid", "sizes"]);
  });

  it("ignores form entries for settings the engine no longer has", () => {
    expect(diffSettings(specs, { ...defaultFormValues(specs), retired_setting: "1" }).changed).toEqual({});
  });

  it("marks changed fields", () => {
    expect(isChanged(spec("grid"), "400")).toBe(false);
    expect(isChanged(spec("grid"), "401")).toBe(true);
    expect(isChanged(spec("grid"), "oops")).toBe(true);
    expect(isChanged(spec("repair_masks"), true)).toBe(false);
    expect(isChanged(spec("repair_masks"), false)).toBe(true);
  });
});

describe("placeServerError", () => {
  const names = [...specs.map((candidate) => candidate.name), "inputs", "reference"];

  it("puts a validation message at the setting it names", () => {
    expect(placeServerError("dense configuration: sizes must increase", names)).toEqual({
      fields: ["sizes"],
      message: "Sizes must increase.",
    });
    expect(placeServerError("dense configuration: grid must be 64..1024", names).fields).toEqual(["grid"]);
  });

  it("marks every setting a combined rule names", () => {
    expect(placeServerError("dense configuration: need 2 <= best_of <= neighbours <= 16", names).fields.sort()).toEqual([
      "best_of",
      "neighbours",
    ]);
  });

  it("places path problems at the path field", () => {
    expect(placeServerError("inputs must be an existing path under the data directory", names).fields).toEqual(["inputs"]);
    expect(placeServerError("reference must be an existing path under the data directory", names).fields).toEqual(["reference"]);
  });

  it("does not match a name inside another word, or the list of known names", () => {
    expect(placeServerError("dense configuration: invalid gridlock", names).fields).toEqual([]);
    expect(placeServerError("unknown setting 'bogus=1'; known: best_of, grid, sizes", names).fields).toEqual([]);
    expect(placeServerError("run did not start: Traceback ...", names).fields).toEqual([]);
  });
});

describe("startRunBody", () => {
  it("leaves out empty optional fields and unchanged settings", () => {
    expect(startRunBody({ name: "", inputs: " sphere ", reference: "", device: "" }, {})).toEqual({ inputs: "sphere" });
  });

  it("matches the contract's start body", () => {
    expect(
      startRunBody({ name: "my dragon", inputs: "dragon/inputs", reference: "dragon/scan.ply", device: "mps" }, { grid: 320, sizes: [256, 512] }),
    ).toEqual({
      name: "my dragon",
      inputs: "dragon/inputs",
      device: "mps",
      settings: { grid: 320, sizes: [256, 512] },
      reference: "dragon/scan.ply",
    });
  });
});
