import { useEffect, useMemo, useState } from "preact/hooks";
import { HttpEngine } from "../sources/httpEngine";
import { ReplaySource } from "../sources/replaySource";
import type { Engine, RunSource } from "../sources/types";
import { Connection } from "./connection";
import { Icon } from "./icons";
import { NewRun } from "./newRun";
import { applyTheme, loadPrefs, savePrefs, speedFromPref, type ThemeChoice } from "./prefs";
import { Runs } from "./runs";
import { RunView } from "./runView";

/** Hash routes, so the app works from any base path and on static hosts without rewrites. */
export type Route =
  | { screen: "connection" }
  | { screen: "replay"; bundle: string | null }
  | { screen: "runs" }
  | { screen: "new" }
  | { screen: "run"; id: string };

export function parseRoute(hash: string): Route {
  const [path = "", query = ""] = hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter((part) => part !== "");
  if (parts[0] === "replay") return { screen: "replay", bundle: new URLSearchParams(query).get("bundle") };
  if (parts[0] === "engine") {
    if (parts[1] === "new") return { screen: "new" };
    if (parts[1] === "run" && parts[2] !== undefined) return { screen: "run", id: decodeURIComponent(parts[2]) };
    return { screen: "runs" };
  }
  return { screen: "connection" };
}

export function navigate(hash: string): void {
  location.hash = hash;
}

const THEME_NEXT: Record<ThemeChoice, ThemeChoice> = { system: "light", light: "dark", dark: "system" };
const THEME_LABEL: Record<ThemeChoice, string> = { system: "Auto", light: "Light", dark: "Dark" };
const THEME_ICON = { system: "auto", light: "sun", dark: "moon" } as const;

/** The demo recording ships next to index.html. */
const DEMO_BUNDLE = "demo/";

export function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(location.hash));
  const [prefs, setPrefs] = useState(loadPrefs);
  const [themeTick, setThemeTick] = useState(0);

  useEffect(() => {
    const onHash = () => setRoute(parseRoute(location.hash));
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    applyTheme(prefs.theme);
    setThemeTick((tick) => tick + 1);
    const media = matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setThemeTick((tick) => tick + 1);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [prefs.theme]);

  const engine = useMemo<Engine | null>(() => {
    if (prefs.engineUrl === "") return null;
    try {
      return new HttpEngine(prefs.engineUrl, { token: prefs.engineToken || undefined });
    } catch {
      return null;
    }
  }, [prefs.engineUrl, prefs.engineToken]);

  const runSource = useMemo<RunSource | null>(() => {
    if (route.screen === "replay") {
      const bundle = route.bundle ?? DEMO_BUNDLE;
      return new ReplaySource(new URL(bundle, document.baseURI).href, { speed: speedFromPref(loadPrefs().replaySpeed) });
    }
    if (route.screen === "run" && engine !== null) return engine.openRun(route.id);
    return null;
    // `route` is a fresh object per navigation; key on what identifies the run.
  }, [route.screen, route.screen === "replay" ? route.bundle : route.screen === "run" ? route.id : "", engine]);

  const update = (change: Partial<typeof prefs>) => setPrefs(savePrefs(change));
  const needsEngine = route.screen === "runs" || route.screen === "new" || route.screen === "run";

  let title = "Studio";
  if (route.screen === "replay") title = route.bundle === null ? "Demo recording" : "Recording";
  else if (route.screen === "runs") title = "Runs";
  else if (route.screen === "new") title = "New run";
  else if (route.screen === "run") title = route.id;
  useEffect(() => {
    document.title = title === "Studio" ? "Crisp3DS Studio" : `${title} - Crisp3DS Studio`;
  }, [title]);

  return (
    <div class="app">
      <a class="skip-link" href="#main">
        Skip to content
      </a>
      <header class="topbar">
        <a class="brand" href="#/" aria-label="Crisp3DS Studio, connection">
          <Icon name="cube" size={20} />
          <span>
            Crisp3DS <strong>Studio</strong>
          </span>
        </a>
        <nav class="topnav" aria-label="Main">
          <a href="#/" aria-current={route.screen === "connection" ? "page" : undefined}>
            Connection
          </a>
          <a href="#/replay" aria-current={route.screen === "replay" && route.bundle === null ? "page" : undefined}>
            Demo
          </a>
          {engine !== null && (
            <a href="#/engine" aria-current={needsEngine ? "page" : undefined}>
              Runs
            </a>
          )}
        </nav>
        <button
          type="button"
          class="button quiet theme-toggle"
          onClick={() => update({ theme: THEME_NEXT[prefs.theme] })}
          aria-label={`Theme: ${THEME_LABEL[prefs.theme]}. Switch to ${THEME_LABEL[THEME_NEXT[prefs.theme]]}.`}
          title="Switch theme"
        >
          <Icon name={THEME_ICON[prefs.theme]} />
          <span class="theme-label">{THEME_LABEL[prefs.theme]}</span>
        </button>
      </header>
      <main id="main" tabIndex={-1}>
        {route.screen === "connection" && <Connection prefs={prefs} onChange={update} />}
        {needsEngine && engine === null && (
          <section class="page narrow">
            <div class="notice bad" role="alert">
              <p>No engine is connected.</p>
              <a class="button primary" href="#/">
                Choose a connection
              </a>
            </div>
          </section>
        )}
        {route.screen === "runs" && engine !== null && <Runs engine={engine} />}
        {route.screen === "new" && engine !== null && <NewRun engine={engine} prefs={prefs} onChange={update} />}
        {runSource !== null && (route.screen === "replay" || route.screen === "run") && (
          <RunView
            key={runSource.kind + runSource.title}
            source={runSource}
            themeTick={themeTick}
            backHref={route.screen === "run" ? "#/engine" : "#/"}
            heading={route.screen === "run" ? route.id : route.bundle === null ? "Demo: synthetic sphere" : "Recorded run"}
          />
        )}
      </main>
    </div>
  );
}
