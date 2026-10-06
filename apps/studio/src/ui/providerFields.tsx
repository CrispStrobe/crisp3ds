import { useEffect, useState } from "preact/hooks";
import { optionError, optionsFor, type OptionValues, type ProviderOptionSpec } from "../core/providerOptions";
import { chosenProviders, importFields, type StartPoint } from "../core/startPoints";
import type { Engine } from "../sources/types";
import { PathField } from "./pathField";

interface Props {
  engine: Engine;
  point: StartPoint;
  /** Chosen provider by module. */
  providers: Record<string, string>;
  onProvider(module: string, id: string): void;
  options: OptionValues;
  onOption(flag: string, value: string | boolean): void;
  values: Record<string, string>;
  onValue(key: string, value: string): void;
  /** Error for a field key (import paths) or an option flag. */
  error(key: string): string | undefined;
  /** Where the tools of unavailable providers are set up, if this app has such a screen. */
  toolsHref?: string;
}

function OptionField({ spec, value, error, onChange }: { spec: ProviderOptionSpec; value: string | boolean | undefined; error?: string; onChange(value: string | boolean): void }) {
  const id = `o-${spec.flag}`;
  const describedBy = [`${id}-help`, error !== undefined ? `${id}-error` : ""].filter(Boolean).join(" ");
  const text = value === undefined ? spec.default : String(value);
  const changed = spec.kind === "boolean" ? (value === undefined ? false : (value === true ? "on" : "off") !== spec.default) : text.trim() !== spec.default;
  return (
    <div class={`field setting${changed ? " changed" : ""}`}>
      {spec.kind === "boolean" ? (
        <label class="check" for={id}>
          <input
            id={id}
            type="checkbox"
            checked={value === undefined ? spec.default === "on" : value === true}
            aria-describedby={describedBy}
            onChange={(event) => onChange(event.currentTarget.checked)}
          />
          <span class="field-label">{spec.label}</span>
        </label>
      ) : (
        <>
          <label class="field-label" for={id}>
            {spec.label}
          </label>
          {spec.kind === "choice" ? (
            <select id={id} value={text} aria-describedby={describedBy} onChange={(event) => onChange(event.currentTarget.value)}>
              {(spec.choices ?? []).map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
            </select>
          ) : (
            <input
              id={id}
              type="text"
              inputMode={spec.kind === "integer" ? "numeric" : spec.kind === "number" ? "decimal" : "text"}
              spellcheck={false}
              autocomplete="off"
              value={text}
              aria-invalid={error !== undefined}
              aria-describedby={describedBy}
              onInput={(event) => onChange(event.currentTarget.value)}
            />
          )}
        </>
      )}
      <span class="help" id={`${id}-help`}>
        {spec.help}
        {changed && spec.default !== "" && ` Default: ${spec.default}.`}
      </span>
      {error !== undefined && (
        <span class="field-error" id={`${id}-error`}>
          {error}
        </span>
      )}
    </div>
  );
}

/**
 * For a start point with interchangeable modules (masks, cameras): which provider does each,
 * with the ones that cannot run shown and explained, and the chosen provider's options.
 */
export function ProviderChoices({ engine, point, providers, onProvider, options, onOption, values, onValue, error, toolsHref }: Props) {
  const chosen = chosenProviders(point, providers);
  const imports = importFields(point, providers);
  return (
    <>
      {point.providers.map((choice) => {
        const current = chosen[choice.module]!;
        const specs = optionsFor(choice.module, current);
        const basic = specs.filter((spec) => !spec.advanced);
        const advanced = specs.filter((spec) => spec.advanced);
        const imported = imports.find((entry) => entry.module === choice.module);
        const changedAdvanced = advanced.filter((spec) => options[spec.flag] !== undefined && String(options[spec.flag] === true ? "on" : options[spec.flag] === false ? "off" : options[spec.flag]).trim() !== spec.default).length;
        return (
          <fieldset class="provider" key={`${point.id}-${choice.module}`}>
            <legend>{choice.label}</legend>
            <div class="provider-list" role="radiogroup" aria-label={choice.label}>
              {choice.options.map((option) => (
                <label key={option.id} class={`provider-option${option.id === current ? " chosen" : ""}${option.available ? "" : " unavailable"}`}>
                  <input
                    type="radio"
                    name={`provider-${choice.module}`}
                    value={option.id}
                    checked={option.id === current}
                    disabled={!option.available}
                    onChange={() => onProvider(choice.module, option.id)}
                  />
                  <span class="provider-text">
                    <span class="provider-name">
                      {option.label}
                      {!option.available && <span class="badge status-cancelled">Not available</span>}
                    </span>
                    {option.meaning !== undefined && <span class="help">{option.meaning}</span>}
                    {option.external.length > 0 && (
                      <span class="help">
                        Needs: {option.external.join(", ")}.{option.license !== undefined && ` License: ${option.license}.`}
                      </span>
                    )}
                    {!option.available && (
                      <span class="field-error">
                        {option.reason ?? "It cannot be used on this computer."}{" "}
                        {toolsHref !== undefined && option.external.length > 0 && <a href={toolsHref}>Tools</a>}
                      </span>
                    )}
                  </span>
                </label>
              ))}
            </div>
            {imported !== undefined && (
              <PathField
                engine={engine}
                want={imported.spec.kind}
                id={`f-${imported.spec.key}`}
                label={imported.spec.label}
                value={values[imported.spec.key] ?? ""}
                onInput={(value) => onValue(imported.spec.key, value)}
                help={imported.spec.help}
                error={error(imported.spec.key)}
                required
              />
            )}
            {basic.length > 0 && (
              <div class="settings-grid">
                {basic.map((spec) => (
                  <OptionField key={spec.flag} spec={spec} value={options[spec.flag]} error={error(spec.flag) ?? optionError(spec, options[spec.flag])} onChange={(value) => onOption(spec.flag, value)} />
                ))}
              </div>
            )}
            {advanced.length > 0 && (
              <details class="raw provider-more">
                <summary>
                  More options for {current}
                  {changedAdvanced > 0 ? ` (${changedAdvanced} changed)` : ""}
                </summary>
                <div class="settings-grid">
                  {advanced.map((spec) => (
                    <OptionField key={spec.flag} spec={spec} value={options[spec.flag]} error={error(spec.flag) ?? optionError(spec, options[spec.flag])} onChange={(value) => onOption(spec.flag, value)} />
                  ))}
                </div>
              </details>
            )}
          </fieldset>
        );
      })}
    </>
  );
}

/** Calibration files the engine knows about, offered under the calibration field. */
export function CalibrationList({ engine, current, onPick }: { engine: Engine; current: string; onPick(path: string): void }) {
  const [found, setFound] = useState<{ label: string; path: string }[]>([]);
  useEffect(() => {
    let alive = true;
    engine.calibrations?.().then(
      (list) => alive && setFound(list),
      () => undefined,
    );
    return () => {
      alive = false;
    };
  }, [engine]);
  if (found.length === 0) return null;
  return (
    <div class="calibrations" role="group" aria-label="Calibrations found on this computer">
      <span class="help">Found on this computer:</span>
      {found.map((calibration) => (
        <button key={calibration.path} type="button" class="button small" aria-pressed={calibration.path === current} title={calibration.path} onClick={() => onPick(calibration.path)}>
          {calibration.label}
        </button>
      ))}
    </div>
  );
}
