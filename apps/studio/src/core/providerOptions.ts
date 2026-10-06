/**
 * The tuning options of the providers of the photos start, and how they become the
 * option words of the photos stage (`crisp3ds-dense photos --help`).
 *
 * The crate describes which providers exist, where they run and what they need, but not
 * (yet) their options as data. Until it does, this table is written by hand from the
 * crate's help text. Options that say where a program or a folder is are deliberately
 * absent: those come from the Tools settings and the form's own fields. No DOM.
 */

export type OptionKind = "text" | "integer" | "number" | "choice" | "boolean";

export interface ProviderOptionSpec {
  /** The flag without dashes, e.g. `alicevision-describer-preset`. Also the key of its value in the form. */
  flag: string;
  label: string;
  kind: OptionKind;
  /** The crate's default, as text ("on"/"off" for booleans). A value equal to it is not sent. */
  default: string;
  choices?: string[];
  help: string;
  /** Shown only after "More options". */
  advanced: boolean;
  /**
   * How a boolean is written: `no-prefix` gives `--flag` / `--no-flag`; `on-off` gives
   * `--flag on` / `--flag off`.
   */
  form?: "no-prefix" | "on-off";
  /** May be given several times: the value is split at spaces into one `--flag word` each. */
  repeat?: boolean;
}

/** A provider that takes its data from a path instead of computing it. */
export interface ImportSpec {
  /** Key in the start request. */
  key: string;
  label: string;
  kind: "folder" | "file";
  help: string;
}

const option = (
  flag: string,
  label: string,
  kind: OptionKind,
  fallback: string,
  help: string,
  more: Partial<ProviderOptionSpec> = {},
): ProviderOptionSpec => ({ flag, label, kind, default: fallback, help, advanced: true, ...more });

/** Options by `module:provider`. Options under `module:*` apply to every provider of the module. */
export const PROVIDER_OPTIONS: Record<string, ProviderOptionSpec[]> = {
  "masks:threshold": [
    option("threshold-level", "Threshold", "text", "otsu", "Grey level below which a pixel is object: a number from 0 to 255, or otsu to choose it per photo.", { advanced: false }),
    option("threshold-envelope", "Search area", "text", "auto", "auto (whole frame) or x0,y0,x1,y1 in pixels; fractions if all are at most 1."),
  ],
  "masks:external-sam": [
    option("sam-device", "Device", "choice", "mps", "Where PyTorch runs SAM.", { choices: ["mps", "cuda", "cpu"], advanced: false }),
    option("sam-multimask", "Several mask candidates", "boolean", "on", "Let SAM propose several masks and keep the best.", { form: "no-prefix" }),
    option("sam-preserve-holes", "Keep holes", "boolean", "on", "Keep holes that SAM leaves inside the object.", { form: "no-prefix" }),
    option("sam-automatic-cues", "Automatic prompts", "boolean", "on", "Prompt SAM from the threshold masks.", { form: "no-prefix" }),
    option("threshold-level", "Threshold for the prompts", "text", "70", "Grey level of the rough masks that prompt SAM (0 to 255, or otsu)."),
    option("sam-timeout", "Time limit, seconds", "integer", "1800", "The whole SAM step is stopped after this long."),
  ],
  "masks:*": [
    option("hole-cleanup-budget", "Hole cleanup budget", "number", "0.02", "Largest share of the foreground that the dark-hole fill may add."),
  ],
  "cameras:alicevision": [
    option("alicevision-describer-preset", "Feature density", "choice", "normal", "How many features AliceVision extracts per photo. Higher is slower and can register more photos.", {
      choices: ["low", "medium", "normal", "high", "ultra"],
      advanced: false,
    }),
    option("alicevision-describer-types", "Feature types", "text", "sift", "Describer types, comma-separated (for example sift,akaze)."),
    option("alicevision-matching-method", "Image matching", "text", "Exhaustive", "AliceVision's imageMatching method."),
    option("alicevision-memory-gib", "Memory for matching, GiB", "number", "4", "Memory AliceVision may use."),
    option("alicevision-initial-field-of-view", "Initial field of view, degrees", "number", "45", "Only used before the declared lens is applied."),
    option("alicevision-sfm-option", "Extra globalSfM arguments", "text", "", "Extra words for aliceVision_globalSfM, separated by spaces.", { repeat: true }),
    option("features-timeout", "Features time limit, seconds", "integer", "1800", "Feature extraction is stopped after this long."),
    option("matching-timeout", "Matching time limit, seconds", "integer", "1800", "Matching is stopped after this long."),
    option("sfm-timeout", "SfM time limit, seconds", "integer", "900", "Camera recovery is stopped after this long."),
  ],
  "cameras:colmap": [
    option("colmap-matching", "Matching", "choice", "exhaustive", "exhaustive: every pair. ring: each photo with its successors around the closed turn. sequential: the same without closing the turn.", {
      choices: ["exhaustive", "ring", "sequential"],
      advanced: false,
    }),
    option("colmap-overlap", "Successors matched", "integer", "10", "For ring and sequential matching."),
    option("colmap-masks", "Features only inside the masks", "boolean", "on", "Ignore the backdrop when detecting features.", { form: "on-off" }),
    option("colmap-max-features", "Features per photo", "integer", "8192", "Upper limit of SIFT features per photo."),
    option("colmap-cli", "COLMAP command line version", "choice", "3", "3: SiftExtraction.* option names. 4: FeatureExtraction.*.", { choices: ["3", "4"] }),
    option("colmap-extractor-option", "Extra feature_extractor arguments", "text", "", "Extra words, separated by spaces.", { repeat: true }),
    option("colmap-matcher-option", "Extra matcher arguments", "text", "", "Extra words, separated by spaces.", { repeat: true }),
    option("colmap-mapper-option", "Extra mapper arguments", "text", "", "Extra words, separated by spaces.", { repeat: true }),
    option("features-timeout", "Features time limit, seconds", "integer", "1800", "Feature extraction is stopped after this long."),
    option("matching-timeout", "Matching time limit, seconds", "integer", "1800", "Matching is stopped after this long."),
    option("sfm-timeout", "Mapper time limit, seconds", "integer", "900", "Camera recovery is stopped after this long."),
  ],
  "cameras:*": [
    option("contrast-gamma", "Contrast gamma", "number", "0.5", "Applied to the images features are detected in; 1 disables it."),
    option("clahe-clip", "Local contrast (CLAHE) clip", "number", "2.0", "0 disables it."),
    option("clahe-grid", "Local contrast grid", "integer", "8", "Tiles per side."),
    option("random-seed", "Random seed", "integer", "0", "For the providers' randomised steps."),
    option("minimum-registered-fraction", "Photos that must be registered", "number", "0.8", "Below this share of the photos the cameras are rejected."),
    option("maximum-view-reprojection-p95", "Largest reprojection error (95th percentile), pixels", "number", "4.0", "Above this in any photo the cameras are rejected."),
  ],
};

export const PROVIDER_IMPORTS: Record<string, ImportSpec> = {
  "masks:import": { key: "masks_import", label: "Masks folder", kind: "folder", help: "One 8-bit PNG per photo, white object, named like the photo or capture_NNNN.png in capture order." },
  "cameras:import": { key: "cameras_import", label: "Camera solution", kind: "file", help: "An AliceVision .sfm file, or a COLMAP model folder (type its path)." },
};

/** The options of one provider: its own, then those every provider of its module has. */
export function optionsFor(module: string, provider: string): ProviderOptionSpec[] {
  const own = PROVIDER_OPTIONS[`${module}:${provider}`] ?? [];
  const shared = (PROVIDER_OPTIONS[`${module}:*`] ?? []).filter((spec) => !own.some((other) => other.flag === spec.flag));
  // An import brings finished data: the options that tune how it would have been computed do not apply,
  // except the cleanup and the gates, which run on imported data too.
  return [...own, ...shared];
}

export type OptionValues = Record<string, string | boolean>;

function normal(spec: ProviderOptionSpec, raw: string | boolean | undefined): string {
  if (raw === undefined) return spec.default;
  if (spec.kind === "boolean") return raw === true || raw === "on" ? "on" : "off";
  return String(raw).trim();
}

/** What is wrong with a value, or undefined. The crate checks again; this only catches typing slips early. */
export function optionError(spec: ProviderOptionSpec, raw: string | boolean | undefined): string | undefined {
  const value = normal(spec, raw);
  if (value === spec.default || spec.kind === "boolean") return undefined;
  if (value === "") return spec.repeat ? undefined : "Needs a value.";
  if (spec.kind === "integer" && !/^[+-]?\d+$/.test(value)) return "Needs a whole number.";
  if (spec.kind === "number" && !Number.isFinite(Number(value))) return "Needs a number.";
  if (spec.kind === "choice" && !(spec.choices ?? []).includes(value)) return `Must be one of: ${(spec.choices ?? []).join(", ")}.`;
  if (!spec.repeat && value.startsWith("--")) return "A value cannot start with two dashes.";
  return undefined;
}

/**
 * The option words for the chosen providers: only values that differ from the crate's
 * defaults, in a stable order, each flag once.
 */
export function optionTokens(chosen: Record<string, string>, values: OptionValues): string[] {
  const tokens: string[] = [];
  const seen = new Set<string>();
  for (const module of Object.keys(chosen).sort()) {
    for (const spec of optionsFor(module, chosen[module]!)) {
      if (seen.has(spec.flag)) continue;
      seen.add(spec.flag);
      const value = normal(spec, values[spec.flag]);
      if (value === spec.default || optionError(spec, values[spec.flag]) !== undefined) continue;
      if (spec.kind === "boolean") {
        if (spec.form === "on-off") tokens.push(`--${spec.flag}`, value);
        else tokens.push(value === "on" ? `--${spec.flag}` : `--no-${spec.flag}`);
      } else if (spec.repeat) {
        for (const word of value.split(/\s+/).filter((part) => part !== "")) tokens.push(`--${spec.flag}`, word);
      } else {
        tokens.push(`--${spec.flag}`, value);
      }
    }
  }
  return tokens;
}

/** Problems by flag, for the chosen providers. */
export function optionErrors(chosen: Record<string, string>, values: OptionValues): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const module of Object.keys(chosen)) {
    for (const spec of optionsFor(module, chosen[module]!)) {
      const problem = optionError(spec, values[spec.flag]);
      if (problem !== undefined) errors[spec.flag] = problem;
    }
  }
  return errors;
}
