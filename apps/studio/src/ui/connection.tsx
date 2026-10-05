import { useEffect, useState } from "preact/hooks";
import { HttpEngine, normaliseEngineUrl } from "../sources/httpEngine";
import { describe } from "../sources/transport";
import { navigate } from "./app";
import { Icon } from "./icons";
import type { Prefs } from "./prefs";
import type { EngineStatus } from "../shell/shell";

interface Props {
  prefs: Prefs;
  onChange(change: Partial<Prefs>): void;
  /** Present inside the desktop app, which runs an engine itself. */
  localEngine: { status: EngineStatus | null; inUse: boolean } | null;
}

const LOCAL_STATE: Record<EngineStatus["state"], string> = {
  stopped: "is stopped",
  starting: "is starting",
  running: "is running",
  failed: "could not be started",
};

export function Connection({ prefs, onChange, localEngine }: Props) {
  const [bundle, setBundle] = useState(prefs.bundleUrl);
  const [url, setUrl] = useState(prefs.engineUrl);
  const [token, setToken] = useState(prefs.engineToken);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [bundleError, setBundleError] = useState("");
  const [sameOrigin, setSameOrigin] = useState(false);

  // If this page is being served by an engine (`engine_server.py --static`), offer it.
  useEffect(() => {
    if (!location.protocol.startsWith("http") || localEngine !== null) return;
    const abort = new AbortController();
    new HttpEngine(location.origin)
      .health(abort.signal)
      .then(() => {
        setSameOrigin(true);
        setUrl((current) => (current === "" ? location.origin : current));
      })
      .catch(() => undefined);
    return () => abort.abort();
  }, []);

  const connect = async (event: Event) => {
    event.preventDefault();
    setError("");
    let engine: HttpEngine;
    let address: string;
    try {
      address = normaliseEngineUrl(url);
      engine = new HttpEngine(address, { token: token || undefined });
    } catch (problem) {
      setError(describe(problem));
      return;
    }
    setBusy(true);
    try {
      await engine.health();
      onChange({ engineUrl: address, engineToken: token, engineChoice: "remote" });
      navigate("#/engine");
    } catch (problem) {
      setError(describe(problem));
    } finally {
      setBusy(false);
    }
  };

  const openBundle = (event: Event) => {
    event.preventDefault();
    const value = bundle.trim();
    if (value === "") {
      setBundleError("Enter the address of a recording.");
      return;
    }
    try {
      new URL(value, document.baseURI);
    } catch {
      setBundleError("That is not a valid address.");
      return;
    }
    onChange({ bundleUrl: value });
    navigate(`#/replay?bundle=${encodeURIComponent(value)}`);
  };

  return (
    <section class="page narrow">
      <h1>Photos to a printable model</h1>
      <p class="lead">
        Studio shows a reconstruction as it happens: the photos, the masks, the depth found at each level and the
        surface getting better step by step. The computing is done by an engine on a desktop computer; this page
        connects to one, or plays back a recorded run.
      </p>

      <div class="cards">
        {localEngine !== null && (
          <section class="card wide" aria-labelledby="c-local">
            <h2 id="c-local">This computer</h2>
            <p>
              The app runs an engine on this computer. It {localEngine.status === null ? "is starting" : LOCAL_STATE[localEngine.status.state]}
              {localEngine.status?.state === "failed" && localEngine.status.message !== null ? `: ${localEngine.status.message}` : "."}
            </p>
            <div class="actions">
              <a class="button primary" href="#/engine" onClick={() => onChange({ engineChoice: "local" })}>
                {localEngine.inUse ? "Open runs" : "Use this computer"}
              </a>
              <a class="button" href="#/shell">
                Engine settings
              </a>
            </div>
          </section>
        )}
        <section class="card" aria-labelledby="c-demo">
          <h2 id="c-demo">Demo recording</h2>
          <p>A recorded run on a synthetic sphere, included with this app. Needs no engine.</p>
          <a class="button primary" href="#/replay">
            <Icon name="play" /> Play the demo
          </a>
        </section>

        <section class="card tall" aria-labelledby="c-engine">
          <h2 id="c-engine">{localEngine !== null ? "Another engine" : "Engine"}</h2>
          <p>
            Connect to a running engine to watch its runs live and start new ones.
            {sameOrigin && " An engine is serving this page."}
          </p>
          <form onSubmit={connect} novalidate>
            <label class="field">
              <span class="field-label">Address</span>
              <input
                type="url"
                inputMode="url"
                autocomplete="url"
                spellcheck={false}
                placeholder="http://127.0.0.1:8765"
                value={url}
                onInput={(event) => setUrl(event.currentTarget.value)}
                aria-describedby={error !== "" ? "engine-error" : undefined}
                aria-invalid={error !== ""}
              />
            </label>
            <label class="field">
              <span class="field-label">
                Access token <span class="optional">only if the engine asks for one</span>
              </span>
              <input
                type="password"
                autocomplete="off"
                spellcheck={false}
                value={token}
                onInput={(event) => setToken(event.currentTarget.value)}
              />
              <span class="help">Kept on this device only.</span>
            </label>
            {error !== "" && (
              <p class="field-error" id="engine-error" role="alert">
                {error}
              </p>
            )}
            <div class="actions">
              <button type="submit" class="button primary" disabled={busy}>
                {busy ? "Connecting..." : "Connect"}
              </button>
              {prefs.engineUrl !== "" && (
                <button
                  type="button"
                  class="button"
                  onClick={() => {
                    onChange({ engineUrl: "", engineToken: "", engineChoice: "local" });
                    setUrl("");
                    setToken("");
                  }}
                >
                  Forget this engine
                </button>
              )}
            </div>
          </form>
        </section>

        <section class="card" aria-labelledby="c-bundle">
          <h2 id="c-bundle">Recorded run</h2>
          <p>Play back a replay bundle: a folder with an event log and the files it names, on any web server.</p>
          <form onSubmit={openBundle} novalidate>
            <label class="field">
              <span class="field-label">Bundle address</span>
              <input
                type="url"
                inputMode="url"
                spellcheck={false}
                placeholder="https://example.org/runs/dragon/"
                value={bundle}
                onInput={(event) => {
                  setBundle(event.currentTarget.value);
                  setBundleError("");
                }}
                aria-invalid={bundleError !== ""}
              />
            </label>
            {bundleError !== "" && (
              <p class="field-error" role="alert">
                {bundleError}
              </p>
            )}
            <div class="actions">
              <button type="submit" class="button">
                Open recording
              </button>
            </div>
          </form>
        </section>
      </div>
    </section>
  );
}
