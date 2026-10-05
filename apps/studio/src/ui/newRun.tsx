import { useEffect, useMemo, useState } from "preact/hooks";
import { humanise } from "../core/format";
import {
  defaultFormValue,
  defaultFormValues,
  diffSettings,
  groupSettings,
  isChanged,
  placeServerError,
  startRunBody,
  type FormValues,
  type SettingSpec,
} from "../core/settings";
import { describe, isAbort } from "../sources/transport";
import { EngineError, type Engine, type EngineHealth } from "../sources/types";
import { navigate } from "./app";
import { Icon } from "./icons";
import type { Prefs } from "./prefs";

interface Props {
  engine: Engine;
  prefs: Prefs;
  onChange(change: Partial<Prefs>): void;
}

const DEVICES = [
  { value: "", label: "Engine default" },
  { value: "mps", label: "Apple GPU (mps)" },
  { value: "cuda", label: "NVIDIA GPU (cuda)" },
  { value: "cpu", label: "CPU" },
];

export function NewRun({ engine, prefs, onChange }: Props) {
  const [specs, setSpecs] = useState<SettingSpec[] | null>(null);
  const [health, setHealth] = useState<EngineHealth | null>(null);
  const [loadError, setLoadError] = useState("");
  const [form, setForm] = useState<FormValues>({});
  const [name, setName] = useState("");
  const [inputs, setInputs] = useState(prefs.inputs);
  const [reference, setReference] = useState("");
  const [device, setDevice] = useState(prefs.device);
  const [busy, setBusy] = useState(false);
  const [serverError, setServerError] = useState<{ fields: string[]; message: string } | null>(null);
  const [showErrors, setShowErrors] = useState(false);

  useEffect(() => {
    const abort = new AbortController();
    engine
      .settings(abort.signal)
      .then((loaded) => {
        setSpecs(loaded);
        setForm(defaultFormValues(loaded));
      })
      .catch((problem) => {
        if (!isAbort(problem)) setLoadError(describe(problem));
      });
    engine
      .health(abort.signal)
      .then(setHealth)
      .catch(() => undefined);
    return () => abort.abort();
  }, [engine]);

  const groups = useMemo(() => groupSettings(specs ?? []), [specs]);
  const diff = useMemo(() => diffSettings(specs ?? [], form), [specs, form]);
  const changedCount = Object.keys(diff.changed).length + Object.keys(diff.errors).length;
  const inputsMissing = inputs.trim() === "";

  const setValue = (settingName: string, value: string | boolean) => {
    setForm((current) => ({ ...current, [settingName]: value }));
    setServerError(null);
  };

  const submit = async (event: Event) => {
    event.preventDefault();
    setShowErrors(true);
    setServerError(null);
    if (inputsMissing || Object.keys(diff.errors).length > 0) {
      // Open the groups that hold a problem and move to the first one.
      requestAnimationFrame(() => {
        const first = document.querySelector<HTMLElement>(".new-run [aria-invalid='true']");
        first?.closest("details")?.setAttribute("open", "");
        first?.focus();
      });
      return;
    }
    setBusy(true);
    try {
      const id = await engine.startRun(startRunBody({ name, inputs, reference, device }, diff.changed));
      onChange({ inputs: inputs.trim(), device });
      navigate(`#/engine/run/${encodeURIComponent(id)}`);
    } catch (problem) {
      const message = describe(problem);
      const names = [...(specs ?? []).map((spec) => spec.name), "inputs", "reference", "device", "name"];
      setServerError(
        problem instanceof EngineError && problem.status === 400 ? placeServerError(message, names) : { fields: [], message },
      );
      requestAnimationFrame(() => document.getElementById("server-error")?.focus());
    } finally {
      setBusy(false);
    }
  };

  const fieldError = (field: string): string | undefined => {
    if (serverError?.fields.includes(field)) return serverError.message;
    return showErrors ? diff.errors[field] : undefined;
  };

  if (loadError !== "") {
    return (
      <section class="page narrow">
        <div class="notice bad" role="alert">
          <p>
            <strong>The settings could not be loaded.</strong> {loadError}
          </p>
          <a class="button" href="#/engine">
            Back to runs
          </a>
        </div>
      </section>
    );
  }

  const inputsError = fieldError("inputs") ?? (showErrors && inputsMissing ? "Say where the inputs are." : undefined);
  const referenceError = fieldError("reference");

  return (
    <section class="page narrow new-run">
      <a class="back-link" href="#/engine">
        <Icon name="back" size={16} /> Runs
      </a>
      <h1>New run</h1>
      <p class="sub">
        Runs start from data that is already on the engine's computer: recovered cameras and one mask per photo.
        Paths are relative to the engine's data directory.
      </p>

      <form onSubmit={submit} novalidate>
        <fieldset class="panel">
          <legend>Data</legend>
          <TextField
            id="f-inputs"
            label="Inputs folder"
            value={inputs}
            onInput={(value) => {
              setInputs(value);
              setServerError(null);
            }}
            placeholder="dragon/inputs"
            help="A folder with cameras.json, the photos and their masks."
            error={inputsError}
            required
          />
          <TextField
            id="f-reference"
            label="Reference scan"
            optional="optional"
            value={reference}
            onInput={(value) => {
              setReference(value);
              setServerError(null);
            }}
            placeholder="dragon/scan.ply"
            help="Used only after the reconstruction, to score the result against an independent scan."
            error={referenceError}
          />
          <div class="field-row">
            <TextField
              id="f-name"
              label="Name"
              optional="optional"
              value={name}
              onInput={setName}
              placeholder="my dragon"
              error={fieldError("name")}
            />
            <label class="field" for="f-device">
              <span class="field-label">Device</span>
              <select id="f-device" value={device} onChange={(event) => setDevice(event.currentTarget.value)}>
                {DEVICES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.value === "" && health?.device !== undefined ? `Engine default (${health.device})` : option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </fieldset>

        <div class="settings-head">
          <h2>Settings</h2>
          <span class="sub" aria-live="polite">
            {changedCount === 0 ? "All at their defaults" : `${changedCount} changed`}
          </span>
          <button
            type="button"
            class="button"
            disabled={changedCount === 0}
            onClick={() => {
              setForm(defaultFormValues(specs ?? []));
              setServerError(null);
            }}
          >
            Reset to defaults
          </button>
        </div>

        {specs === null && <p class="sub">Loading settings...</p>}

        {groups.map((group, index) => {
          const changedHere = group.settings.filter((spec) => isChanged(spec, form[spec.name])).length;
          return (
            <details class="panel settings-group" key={group.group} open={index === 0}>
              <summary>
                <span class="group-name">{group.group}</span>
                <span class="sub">
                  {group.settings.length} settings{changedHere > 0 ? `, ${changedHere} changed` : ""}
                </span>
              </summary>
              <div class="settings-grid">
                {group.settings.map((spec) => (
                  <SettingField
                    key={spec.name}
                    spec={spec}
                    value={form[spec.name] ?? defaultFormValue(spec)}
                    error={fieldError(spec.name)}
                    onChange={(value) => setValue(spec.name, value)}
                  />
                ))}
              </div>
            </details>
          );
        })}

        {serverError !== null && (
          <div class="notice bad" role="alert" id="server-error" tabIndex={-1}>
            <p>
              <strong>The engine did not accept this run.</strong> {serverError.message}
            </p>
          </div>
        )}

        <div class="actions sticky-actions">
          <button type="submit" class="button primary" disabled={busy || specs === null}>
            {busy ? "Starting..." : "Start run"}
          </button>
          <a class="button" href="#/engine">
            Cancel
          </a>
        </div>
      </form>
    </section>
  );
}

interface TextFieldProps {
  id: string;
  label: string;
  optional?: string;
  value: string;
  placeholder?: string;
  help?: string;
  error?: string;
  required?: boolean;
  onInput(value: string): void;
}

function TextField({ id, label, optional, value, placeholder, help, error, required, onInput }: TextFieldProps) {
  const describedBy = [help !== undefined ? `${id}-help` : "", error !== undefined ? `${id}-error` : ""].filter(Boolean).join(" ");
  return (
    <div class="field">
      <label class="field-label" for={id}>
        {label} {optional !== undefined && <span class="optional">{optional}</span>}
      </label>
      <input
        id={id}
        type="text"
        spellcheck={false}
        autocapitalize="off"
        autocomplete="off"
        value={value}
        placeholder={placeholder}
        required={required}
        aria-invalid={error !== undefined}
        aria-describedby={describedBy || undefined}
        onInput={(event) => onInput(event.currentTarget.value)}
      />
      {help !== undefined && (
        <span class="help" id={`${id}-help`}>
          {help}
        </span>
      )}
      {error !== undefined && (
        <span class="field-error" id={`${id}-error`}>
          {error}
        </span>
      )}
    </div>
  );
}

interface SettingFieldProps {
  spec: SettingSpec;
  value: string | boolean;
  error?: string;
  onChange(value: string | boolean): void;
}

function SettingField({ spec, value, error, onChange }: SettingFieldProps) {
  const id = `s-${spec.name}`;
  const changed = isChanged(spec, value);
  const fallback = defaultFormValue(spec);
  const describedBy = [`${id}-help`, error !== undefined ? `${id}-error` : ""].filter(Boolean).join(" ");
  const list = spec.kind === "integer_list" || spec.kind === "number_list";
  const numeric = spec.kind === "integer" || spec.kind === "number";
  return (
    <div class={`field setting${changed ? " changed" : ""}`}>
      {spec.kind === "boolean" ? (
        <label class="check" for={id}>
          <input
            id={id}
            type="checkbox"
            checked={value === true}
            aria-describedby={describedBy}
            aria-invalid={error !== undefined}
            onChange={(event) => onChange(event.currentTarget.checked)}
          />
          <span class="field-label">{humanise(spec.name)}</span>
        </label>
      ) : (
        <>
          <label class="field-label" for={id}>
            {humanise(spec.name)}
          </label>
          <input
            id={id}
            type="text"
            inputMode={spec.kind === "integer" ? "numeric" : numeric ? "decimal" : "text"}
            spellcheck={false}
            autocomplete="off"
            value={String(value)}
            aria-describedby={describedBy}
            aria-invalid={error !== undefined}
            onInput={(event) => onChange(event.currentTarget.value)}
          />
        </>
      )}
      <span class="help" id={`${id}-help`}>
        {spec.meaning}
        {spec.meaning !== "" && !/[.!?]$/.test(spec.meaning) && "."}
        {list && " Several values, separated by commas."}
        {changed && (
          <>
            {" "}
            <button type="button" class="link" onClick={() => onChange(fallback)}>
              Default: {typeof fallback === "boolean" ? (fallback ? "on" : "off") : fallback}
            </button>
          </>
        )}
      </span>
      {error !== undefined && (
        <span class="field-error" id={`${id}-error`}>
          {error}
        </span>
      )}
    </div>
  );
}
