/**
 * How a run may start, as data. An engine says which starting points it offers; the
 * "New run" form is built from that list. A start from plain photos, with a choice of
 * provider per module (masks, cameras; see docs/ARCHITECTURE.md), is one more entry with
 * `providers` filled in: nothing in the form has to change for it. No DOM.
 */

import { optionTokens, PROVIDER_IMPORTS, type ImportSpec, type OptionValues } from "./providerOptions";
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
  /** False when it cannot be used right now; `reason` says why. Absent means usable. */
  available: boolean;
  reason?: string;
  /** External programs it starts; empty when everything runs inside the engine. */
  external: string[];
  license?: string;
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
  {
    id: "photos",
    label: "Turntable photos",
    meaning: "A folder of photos of an object on a turntable and the calibration of the lens; the engine makes masks and cameras first, with the tools installed on its computer.",
    fields: [
      { key: "photos", label: "Photos", kind: "folder", required: true, help: "One photo per turntable position, named in capture order." },
      { key: "calibration", label: "Lens calibration (.json)", kind: "file", required: true, help: "Focal length, principal point and radial distortion of the lens." },
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
        if (optionId === undefined) continue;
        options.push({
          id: optionId,
          label: text(option.label) ?? optionId,
          meaning: text(option.meaning),
          available: option.available !== false,
          reason: text(option.reason),
          external: Array.isArray(option.external) ? option.external.filter((item): item is string => typeof item === "string") : [],
          license: text(option.license),
        });
      }
      if (module === undefined || options.length === 0) continue;
      const preferred = text(choice.default);
      providers.push({
        module,
        label: text(choice.label) ?? module,
        options,
        // The engine's default if it can be used, else the first provider that can, else the default anyway.
        default: (options.find((option) => option.id === preferred && option.available) ?? options.find((option) => option.available) ?? options.find((option) => option.id === preferred) ?? options[0]!).id,
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
  /** Tuning options of the chosen providers, by flag. */
  options?: OptionValues;
}

/** The provider in effect for each module of a start point: the chosen one if offered, else the default. */
export function chosenProviders(point: StartPoint, picked: Record<string, string>): Record<string, string> {
  const chosen: Record<string, string> = {};
  for (const choice of point.providers) {
    const wanted = picked[choice.module];
    chosen[choice.module] = choice.options.some((option) => option.id === wanted) ? wanted! : (choice.default ?? choice.options[0]!.id);
  }
  return chosen;
}

/** Keys of the paths that the chosen providers import their data from. */
export function importFields(point: StartPoint, picked: Record<string, string>): { module: string; spec: ImportSpec }[] {
  const chosen = chosenProviders(point, picked);
  return Object.entries(chosen).flatMap(([module, provider]) => {
    const spec = PROVIDER_IMPORTS[`${module}:${provider}`];
    return spec === undefined ? [] : [{ module, spec }];
  });
}

/** Required fields of the start point that are still empty, by key. */
export function missingFields(point: StartPoint, values: Record<string, string>, picked: Record<string, string> = {}): string[] {
  const missing = point.fields.filter((field) => field.required && (values[field.key] ?? "").trim() === "").map((field) => field.key);
  for (const { spec } of importFields(point, picked)) if ((values[spec.key] ?? "").trim() === "") missing.push(spec.key);
  return missing;
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
    const providers = chosenProviders(point, form.providers);
    body.providers = providers;
    for (const { spec } of importFields(point, form.providers)) {
      const value = (form.values[spec.key] ?? "").trim();
      if (value !== "") body[spec.key] = value;
    }
    const tokens = optionTokens(providers, form.options ?? {});
    if (tokens.length > 0) body.photo_options = tokens;
  }
  if (Object.keys(changed).length > 0) body.settings = changed;
  return body;
}
