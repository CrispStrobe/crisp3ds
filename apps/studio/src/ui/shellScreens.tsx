import { useEffect, useState } from "preact/hooks";
import {
  sameConfig,
  sourceText,
  type ConfigField,
  type EngineStatus,
  type Shell,
  type ShellConfig,
  type ShellInfo,
  type ShellSettings,
} from "../shell/shell";
import { describe } from "../sources/transport";

/** Polls the shell for the local engine's state: quickly while it starts, lazily once it runs. */
export function useLocalEngine(shell: Shell | null): { info: ShellInfo | null; status: EngineStatus | null; refresh(): void } {
  const [info, setInfo] = useState<ShellInfo | null>(null);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (shell === null) return;
    shell.info().then(setInfo, () => undefined);
  }, [shell]);
  const canRun = info?.can_run_engine === true;
  const state = status?.state;
  useEffect(() => {
    if (shell === null || !canRun) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ask = () => {
      shell.engineStatus().then(
        (next) => {
          if (!alive) return;
          setStatus((previous) => (previous !== null && JSON.stringify(previous) === JSON.stringify(next) ? previous : next));
          timer = setTimeout(ask, next.state === "starting" ? 250 : 2500);
        },
        () => {
          if (alive) timer = setTimeout(ask, 2500);
        },
      );
    };
    ask();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
    // Re-arm when the state changes, so the polling pace follows it at once.
  }, [shell, canRun, state, tick]);
  return { info, status, refresh: () => setTick((count) => count + 1) };
}

export const NOT_BUNDLED =
  "This app does not contain Python or PyTorch. It runs the reconstruction with the interpreters named here, from a crisp3ds source folder on this computer.";

const STATE_TEXT: Record<EngineStatus["state"], string> = {
  stopped: "Stopped",
  starting: "Starting",
  running: "Running",
  failed: "Not running",
};

export function EngineLog({ status }: { status: EngineStatus }) {
  if (status.command === null && status.log.length === 0) return null;
  return (
    <details class="raw" open={status.state === "failed"}>
      <summary>What was started, and what it printed</summary>
      {status.command !== null && (
        <pre class="error-text" tabIndex={0}>
          {status.command}
        </pre>
      )}
      <pre class="error-text" tabIndex={0}>
        {status.log.length > 0 ? status.log.slice(-40).join("\n") : "(nothing was printed)"}
      </pre>
    </details>
  );
}

/** Shown instead of a screen that needs the local engine while it is not there. */
export function LocalEngineGate({ status, shell, onChange }: { status: EngineStatus | null; shell: Shell; onChange(): void }) {
  if (status === null || status.state === "starting" || status.state === "stopped") {
    return (
      <section class="page narrow">
        <p class="sub" role="status">
          Starting the engine on this computer...
        </p>
      </section>
    );
  }
  return (
    <section class="page narrow">
      <div class="notice bad" role="alert">
        <p>
          <strong>The engine on this computer could not be started.</strong> {status.message}
        </p>
        <EngineLog status={status} />
        <div class="actions">
          <a class="button primary" href="#/shell">
            Engine settings
          </a>
          <button
            type="button"
            class="button"
            onClick={() => {
              void shell.restartEngine().then(onChange);
            }}
          >
            Try again
          </button>
          <a class="button" href="#/">
            Other connections
          </a>
        </div>
      </div>
      <p class="sub">{NOT_BUNDLED}</p>
    </section>
  );
}

interface FieldSpec {
  field: ConfigField;
  label: string;
  help: string;
  pick?: "folder" | "file";
  pickTitle?: string;
}

const FIELDS: FieldSpec[] = [
  {
    field: "repo",
    label: "Crisp3DS source folder",
    help: "Your checkout of the crisp3ds repository: the folder that contains scripts/turntable_mesh.",
    pick: "folder",
    pickTitle: "Choose the crisp3ds source folder",
  },
  {
    field: "python",
    label: "Python",
    help: "Interpreter with NumPy, SciPy, scikit-image, OpenCV and Pillow. It also runs the engine itself.",
    pick: "file",
    pickTitle: "Choose the Python interpreter",
  },
  {
    field: "torch_python",
    label: "Python with PyTorch",
    help: "Interpreter for the stereo stage. Leave empty if the one above has PyTorch too.",
    pick: "file",
    pickTitle: "Choose the Python interpreter that has PyTorch",
  },
  {
    field: "data_dir",
    label: "Data folder",
    help: "Runs can only start from folders inside this one. Put your prepared photo sets here.",
    pick: "folder",
    pickTitle: "Choose the data folder",
  },
  {
    field: "runs_dir",
    label: "Runs folder",
    help: "Every run gets a folder in here, with its event log, sheets and meshes.",
    pick: "folder",
    pickTitle: "Choose the folder for runs",
  },
];

export function ShellSettingsScreen({ shell, info, status, onChange }: { shell: Shell; info: ShellInfo | null; status: EngineStatus | null; onChange(): void }) {
  const [settings, setSettings] = useState<ShellSettings | null>(null);
  const [form, setForm] = useState<ShellConfig | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    shell.settings().then(
      (loaded) => {
        setSettings(loaded);
        setForm(loaded.saved);
      },
      (problem) => setError(describe(problem)),
    );
  }, [shell]);

  if (error !== "" && settings === null) {
    return (
      <section class="page narrow">
        <div class="notice bad" role="alert">
          <p>
            <strong>The settings could not be read.</strong> {error}
          </p>
        </div>
      </section>
    );
  }
  if (settings === null || form === null) return null;

  const change = (field: ConfigField, value: string) => setForm({ ...form, [field]: value });
  const dirty = !sameConfig(form, settings.saved);

  const apply = async (event: Event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const saved = await shell.saveSettings(form);
      setSettings(saved);
      setForm(saved.saved);
      await shell.restartEngine();
      onChange();
    } catch (problem) {
      setError(describe(problem));
    } finally {
      setBusy(false);
    }
  };

  const pick = async (spec: FieldSpec) => {
    if (spec.pick === undefined) return;
    const current = form[spec.field] || settings.resolved[spec.field].value;
    const chosen = await shell.pickPath(spec.pick, spec.pickTitle ?? spec.label, current || undefined).catch(() => null);
    if (chosen !== null) change(spec.field, chosen);
  };

  return (
    <section class="page narrow">
      <h1>Engine on this computer</h1>
      <div class="notice" role="note">
        <p>
          <strong>Python is not part of this app.</strong> {NOT_BUNDLED} See the README for what to install.
        </p>
      </div>

      <section class="panel" aria-labelledby="shell-state">
        <div class="panel-head">
          <h2 id="shell-state">State</h2>
          <span class={`badge status-${status?.state === "running" ? "complete" : status?.state === "failed" ? "failed" : "running"}`} role="status">
            {status === null ? "Unknown" : STATE_TEXT[status.state]}
          </span>
        </div>
        {status?.state === "running" && (
          <p class="sub">
            Listening on <span class="mono">{status.url}</span>, this computer only, device {status.device}.{" "}
            <a href="#/engine">Open runs</a>
          </p>
        )}
        {status?.state === "failed" && (
          <p class="field-error" role="alert">
            {status.message}
          </p>
        )}
        {status !== null && <EngineLog status={status} />}
      </section>

      <form onSubmit={apply} novalidate>
        <fieldset class="panel">
          <legend>Where things are</legend>
          {FIELDS.map((spec) => {
            const resolved = settings.resolved[spec.field];
            const id = `shell-${spec.field}`;
            return (
              <div class="field" key={spec.field}>
                <label class="field-label" for={id}>
                  {spec.label}
                </label>
                <div class="path-row">
                  <input
                    id={id}
                    type="text"
                    spellcheck={false}
                    autocapitalize="off"
                    autocomplete="off"
                    value={form[spec.field]}
                    placeholder={resolved.source === "setting" ? "" : resolved.value || "not found"}
                    aria-describedby={`${id}-help`}
                    onInput={(event) => change(spec.field, event.currentTarget.value)}
                  />
                  {spec.pick !== undefined && (
                    <button type="button" class="button" onClick={() => void pick(spec)}>
                      Choose<span class="visually-hidden"> {spec.label}</span>
                    </button>
                  )}
                </div>
                <span class="help" id={`${id}-help`}>
                  {spec.help}
                  {form[spec.field].trim() === "" && (
                    <>
                      {" "}
                      {resolved.value === ""
                        ? "Not set, and nothing was found."
                        : `Not set here: the value shown in the field is used (${sourceText(resolved.source, spec.field)}).`}
                    </>
                  )}
                </span>
              </div>
            );
          })}
          <label class="field" for="shell-device">
            <span class="field-label">Device</span>
            <select id="shell-device" value={form.device === "" ? "auto" : form.device} onChange={(event) => change("device", event.currentTarget.value)}>
              <option value="auto">Automatic{info !== null ? ` (${info.auto_device} on this computer)` : ""}</option>
              <option value="mps">Apple GPU (mps)</option>
              <option value="cuda">NVIDIA GPU (cuda)</option>
              <option value="cpu">CPU</option>
            </select>
            <span class="help">The default for new runs; each run can choose another.</span>
          </label>
        </fieldset>

        {error !== "" && (
          <div class="notice bad" role="alert">
            <p>{error}</p>
          </div>
        )}
        <div class="actions">
          <button type="submit" class="button primary" disabled={busy}>
            {dirty ? "Save and restart the engine" : "Restart the engine"}
          </button>
          <a class="button" href="#/engine">
            Runs
          </a>
        </div>
        <p class="help">
          Saved in <span class="mono wrap">{settings.file}</span>.
        </p>
      </form>
    </section>
  );
}
