/**
 * How a run may start, as data. An engine says which starting points it offers; the
 * "New run" form is built from that list. A start from plain photos, with a choice of
 * provider per module (masks, cameras; see docs/ARCHITECTURE.md), is one more entry with
 * `providers` filled in: nothing in the form has to change for it. No DOM.
 */

import type { SettingValue } from "./settings";

export interface StartField {
  /** Key in the start request (`inputs`, `scene`, `prepared`, `raw_masks`, later `photos`, ...). */
  key: string;
  label: string;
  /** `inputs`: a folder a run can start from. `folder`, `file`: any folder or file. */
  kind: "inputs" | "folder" | "file";
  required: boolean;
  help?: string;
}

export interface ProviderOption {
  id: string;
  label: string;
  meaning?: string;
}

/** One module of the pipeline with interchangeable implementations. */
export interface ProviderChoice {
  /** `masks`, `cameras`, ...: also the key under `providers` in the start request. */
  module: string;
  label: string;
  options: ProviderOption[];
  default?: string;
}

export interface StartPoint {
  id: string;
  label: string;
  meaning?: string;
  fields: StartField[];
  providers: ProviderChoice[];
}

/** What every engine of docs/ENGINE-CONTRACT.md accepts: an inputs folder, or scene + prepared + raw masks. */
export const CONTRACT_START_POINTS: StartPoint[] = [
  {
    id: "inputs",
    label: "Inputs folder",
    meaning: "A folder with cameras.json, the undistorted photos and their masks.",
    fields: [{ key: "inputs", label: "Inputs folder", kind: "inputs", required: true, help: "A folder with cameras.json, the photos and their masks." }],
    providers: [],
  },
  {
    id: "scene",
    label: "Camera solution, images and masks",
    meaning: "An AliceVision solution with its prepared images and one raw mask per photo.",
    fields: [
      { key: "scene", label: "Camera solution (.sfm)", kind: "file", required: true, help: "AliceVision SfM file with poses and one radialk3 lens." },
      { key: "prepared", label: "Prepared images", kind: "folder", required: true, help: "Undistorted images named <viewId>.png." },
      { key: "raw_masks", label: "Raw masks", kind: "folder", required: true, help: "One 0/255 mask per source photo, named like the photo." },
    ],
    providers: [],
  },
];

function text(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function rows(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((row): row is Record<string, unknown> => typeof row === "object" && row !== null) : [];
}

/** Reads a start-point list given as JSON, dropping what it cannot use. Never throws. */
export function parseStartPoints(json: unknown): StartPoint[] {
  const points: StartPoint[] = [];
  for (const row of rows(json)) {
    const id = text(row.id);
    if (id === undefined) continue;
    const fields: StartField[] = [];
    for (const field of rows(row.fields)) {
      const key = text(field.key);
      if (key === undefined) continue;
      const kind = field.kind === "inputs" || field.kind === "file" ? field.kind : "folder";
      fields.push({ key, label: text(field.label) ?? key, kind, required: field.required !== false, help: text(field.help) });
    }
    const providers: ProviderChoice[] = [];
    for (const choice of rows(row.providers)) {
      const module = text(choice.module);
      const options: ProviderOption[] = [];
      for (const option of rows(choice.options)) {
        const optionId = text(option.id);
        if (optionId !== undefined) options.push({ id: optionId, label: text(option.label) ?? optionId, meaning: text(option.meaning) });
      }
      if (module === undefined || options.length === 0) continue;
      const preferred = text(choice.default);
      providers.push({
        module,
        label: text(choice.label) ?? module,
        options,
        default: options.some((option) => option.id === preferred) ? preferred : options[0]!.id,
      });
    }
    if (fields.length === 0) continue;
    points.push({ id, label: text(row.label) ?? id, meaning: text(row.meaning), fields, providers });
  }
  return points;
}

export interface StartForm {
  name: string;
  device: string;
  reference: string;
  /** Field values by key, for the chosen start point. */
  values: Record<string, string>;
  /** Chosen provider by module. */
  providers: Record<string, string>;
}

/** Required fields of the start point that are still empty, by key. */
export function missingFields(point: StartPoint, values: Record<string, string>): string[] {
  return point.fields.filter((field) => field.required && (values[field.key] ?? "").trim() === "").map((field) => field.key);
}

/**
 * Body of `POST /api/runs` for one start point: only that start point's fields, empty
 * optional values left out, providers only where the start point has a choice, and only
 * the settings that differ from their defaults.
 */
export function startBody(point: StartPoint, form: StartForm, changed: Record<string, SettingValue>): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const field of point.fields) {
    const value = (form.values[field.key] ?? "").trim();
    if (value !== "") body[field.key] = value;
  }
  if (form.name.trim() !== "") body.name = form.name.trim();
  if (form.device.trim() !== "") body.device = form.device.trim();
  if (form.reference.trim() !== "") body.reference = form.reference.trim();
  if (point.providers.length > 0) {
    const providers: Record<string, string> = {};
    for (const choice of point.providers) {
      const chosen = form.providers[choice.module];
      providers[choice.module] = choice.options.some((option) => option.id === chosen) ? chosen! : (choice.default ?? choice.options[0]!.id);
    }
    body.providers = providers;
  }
  if (Object.keys(changed).length > 0) body.settings = changed;
  return body;
}
