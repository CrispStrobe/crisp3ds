import { useEffect, useRef, useState } from "preact/hooks";
import { formatBytes, formatCount } from "../core/format";
import { stlBytes, type MeshStep, type RunStatus } from "../core/reducer";
import { describe, isAbort } from "../sources/transport";
import type { RunSource } from "../sources/types";
import { MeshLoader, type LoadProgress } from "../viewer/meshLoader";
import type { UpAxis, Viewer } from "../viewer/viewer";
import { Icon } from "./icons";
import { loadPrefs, savePrefs } from "./prefs";

interface Props {
  source: RunSource;
  meshes: MeshStep[];
  runStatus: RunStatus;
  themeTick: number;
}

/** Meshes above this size are downloaded only when asked for (the final STL can be 50 MB). */
export const AUTO_LOAD_BYTES = 12 * 1024 * 1024;

const UP_CHOICES: UpAxis[] = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"];

type Loading = { path: string; progress: LoadProgress } | null;

export function MeshPanel({ source, meshes, runStatus, themeTick }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewer = useRef<Viewer | null>(null);
  const loader = useRef<MeshLoader | null>(null);
  const strip = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [unsupported, setUnsupported] = useState("");
  /** Path chosen by hand, or null to follow the newest surface. */
  const [chosen, setChosen] = useState<string | null>(null);
  const [approved, setApproved] = useState<ReadonlySet<string>>(new Set());
  const [shown, setShown] = useState<{ path: string; triangles: number } | null>(null);
  const [loading, setLoading] = useState<Loading>(null);
  const [error, setError] = useState<{ path: string; message: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const [flat, setFlat] = useState(() => loadPrefs().flat);
  const [up, setUp] = useState<UpAxis>(() => {
    const stored = loadPrefs().up as UpAxis;
    return UP_CHOICES.includes(stored) ? stored : "+Y";
  });

  // The viewer (and three.js with it) is loaded on demand, so the rest of the page is up first.
  useEffect(() => {
    let cancelled = false;
    loader.current = new MeshLoader(source);
    void import("../viewer/viewer")
      .then((module) => {
        if (cancelled || canvas.current === null || host.current === null) return;
        if (!module.webglAvailable()) {
          setUnsupported("This browser cannot show 3D (WebGL is not available). Everything else still works.");
          return;
        }
        viewer.current = new module.Viewer(canvas.current, host.current);
        setReady(true);
      })
      .catch((problem) => {
        if (!cancelled) setUnsupported(`The 3D view could not be started: ${describe(problem)}`);
      });
    return () => {
      cancelled = true;
      viewer.current?.dispose();
      viewer.current = null;
      loader.current?.dispose();
      loader.current = null;
    };
  }, [source]);

  useEffect(() => {
    if (!ready) return;
    const style = getComputedStyle(document.documentElement);
    viewer.current?.setColors({
      background: style.getPropertyValue("--viewer-bg").trim() || "#e9ebee",
      surface: style.getPropertyValue("--viewer-surface").trim() || "#b9bec6",
    });
  }, [ready, themeTick]);

  useEffect(() => {
    viewer.current?.setFlatShading(flat);
  }, [ready, flat]);

  useEffect(() => {
    viewer.current?.setUp(up);
  }, [ready, up]);

  const newest = meshes[meshes.length - 1];
  const selected = (chosen !== null ? meshes.find((step) => step.path === chosen) : undefined) ?? newest;
  const following = chosen === null || selected === undefined || selected.path !== chosen;

  const needsConsent = (step: MeshStep): boolean => {
    if (step.kind !== "final_mesh" || approved.has(step.path) || loader.current?.has(step.path)) return false;
    const bytes = stlBytes(step.triangles);
    return bytes === undefined || bytes > AUTO_LOAD_BYTES;
  };

  // What to put on screen: the selected step, unless it is a big final mesh nobody asked for
  // yet. Then the newest smaller surface stays up (when following) under a "load" button.
  const awaitingConsent = selected !== undefined && needsConsent(selected);
  let target: MeshStep | undefined = selected;
  if (awaitingConsent) {
    target = following ? [...meshes].reverse().find((step) => !needsConsent(step)) : undefined;
  }
  const targetPath = target?.path;

  useEffect(() => {
    if (!ready || viewer.current === null || loader.current === null) return;
    if (targetPath === undefined) {
      if (meshes.length === 0) {
        viewer.current.clearMesh();
        setShown(null);
      }
      return;
    }
    if (shown?.path === targetPath && viewer.current.hasMesh()) return;
    const abort = new AbortController();
    setError(null);
    setLoading({ path: targetPath, progress: { phase: "download" } });
    loader.current
      .load(targetPath, abort.signal, (progress) => {
        if (!abort.signal.aborted) setLoading({ path: targetPath, progress });
      })
      .then((mesh) => {
        if (abort.signal.aborted || viewer.current === null) return;
        // The previous mesh's GPU buffers are released inside setMesh.
        viewer.current.setMesh(mesh);
        setShown({ path: targetPath, triangles: mesh.triangles - mesh.dropped });
        setLoading(null);
      })
      .catch((problem) => {
        if (abort.signal.aborted || isAbort(problem)) return;
        setLoading(null);
        setError({ path: targetPath, message: describe(problem) });
      });
    return () => abort.abort();
    // `shown` is deliberately not a dependency: it is the result of this effect.
  }, [ready, targetPath, meshes.length === 0, retry]);

  // Keep the selected step visible in the strip.
  useEffect(() => {
    const element = strip.current?.querySelector<HTMLElement>("[aria-pressed='true']");
    const container = strip.current;
    if (element === null || element === undefined || container === null) return;
    const left = element.offsetLeft - container.offsetLeft;
    if (left < container.scrollLeft || left + element.offsetWidth > container.scrollLeft + container.clientWidth) {
      container.scrollTo({ left: left - 16, behavior: "auto" });
    }
  }, [selected?.path, meshes.length]);

  const selectedIndex = selected === undefined ? -1 : meshes.indexOf(selected);
  const select = (index: number) => {
    const step = meshes[index];
    if (step !== undefined) setChosen(step.path);
  };
  const shownStep = shown === null ? undefined : meshes.find((step) => step.path === shown.path);
  const consentBytes = selected !== undefined ? stlBytes(selected.triangles) : undefined;
  const loadingStep = loading === null ? undefined : meshes.find((step) => step.path === loading.path);

  return (
    <section class="panel mesh-panel" aria-labelledby="mesh-heading">
      <div class="panel-head">
        <h2 id="mesh-heading">Surface</h2>
        <span class="sub" aria-live="polite">
          {shownStep !== undefined && shown !== null
            ? `${shownStep.label} · ${formatCount(shown.triangles)} triangles`
            : meshes.length === 0
              ? ""
              : "No surface loaded"}
        </span>
      </div>

      <div class="viewer" ref={host}>
        <canvas
          ref={canvas}
          tabIndex={0}
          role="img"
          aria-label={
            shownStep !== undefined
              ? `3D view: ${shownStep.label}. Drag or use the arrow keys to turn, scroll, pinch or press plus and minus to zoom, right-drag or two fingers to move, 0 to reset.`
              : "3D view, empty"
          }
          onKeyDown={(event) => {
            const view = viewer.current;
            if (view === null || event.ctrlKey || event.metaKey || event.altKey) return;
            const turn = Math.PI / 24;
            if (event.key === "ArrowLeft") view.orbit(-turn, 0);
            else if (event.key === "ArrowRight") view.orbit(turn, 0);
            else if (event.key === "ArrowUp") view.orbit(0, -turn);
            else if (event.key === "ArrowDown") view.orbit(0, turn);
            else if (event.key === "+" || event.key === "=") view.zoom(0.85);
            else if (event.key === "-") view.zoom(1 / 0.85);
            else if (event.key === "0" || event.key === "Home") view.resetView();
            else return;
            event.preventDefault();
          }}
        />
        {unsupported !== "" && <p class="viewer-message">{unsupported}</p>}
        {unsupported === "" && meshes.length === 0 && (
          <p class="viewer-message">
            {runStatus === "running" || runStatus === "waiting"
              ? "The first surface appears here once the silhouette hull is built."
              : "This run produced no surface."}
          </p>
        )}
        {awaitingConsent && selected !== undefined && loading === null && (
          <div class="viewer-overlay">
            <p>
              <strong>{selected.label}</strong> is ready
              {selected.triangles !== undefined && <> · {formatCount(selected.triangles)} triangles</>}
            </p>
            <button
              type="button"
              class="button primary"
              onClick={() => setApproved((current) => new Set(current).add(selected.path))}
            >
              Load {selected.label.toLowerCase()}
              {consentBytes !== undefined && ` (${formatBytes(consentBytes)})`}
            </button>
          </div>
        )}
        {loading !== null && (
          <div class="viewer-status" role="status">
            {loading.progress.phase === "parse"
              ? `Preparing ${loadingStep?.label ?? "surface"}...`
              : `Loading ${loadingStep?.label ?? "surface"}${
                  loading.progress.received !== undefined
                    ? ` · ${formatBytes(loading.progress.received)}${
                        loading.progress.total !== undefined ? ` of ${formatBytes(loading.progress.total)}` : ""
                      }`
                    : "..."
                }`}
          </div>
        )}
        {error !== null && (
          <div class="viewer-overlay" role="alert">
            <p>
              <strong>The surface could not be loaded.</strong> {error.message}
            </p>
            <button type="button" class="button" onClick={() => setRetry((count) => count + 1)}>
              Try again
            </button>
          </div>
        )}
      </div>

      <div class="step-bar">
        <button
          type="button"
          class="button icon-only"
          onClick={() => select(selectedIndex - 1)}
          disabled={selectedIndex <= 0}
          aria-label="Earlier surface"
        >
          <Icon name="prev" />
        </button>
        <div class="step-strip" ref={strip} role="group" aria-label="Surfaces, oldest first">
          {meshes.map((step, index) => (
            <button
              key={step.path}
              type="button"
              class="step"
              aria-pressed={step === selected}
              onClick={() => select(index)}
            >
              <span class="step-index">{index + 1}</span>
              <span class="step-label">{step.label}</span>
              {step.triangles !== undefined && <span class="step-meta">{formatCount(step.triangles)} triangles</span>}
            </button>
          ))}
          {meshes.length === 0 && <span class="sub step-empty">No surfaces yet</span>}
        </div>
        <button
          type="button"
          class="button icon-only"
          onClick={() => select(selectedIndex + 1)}
          disabled={selectedIndex < 0 || selectedIndex >= meshes.length - 1}
          aria-label="Later surface"
        >
          <Icon name="next" />
        </button>
      </div>

      <div class="viewer-tools">
        <label class="check">
          <input type="checkbox" checked={following} onChange={(event) => setChosen(event.currentTarget.checked ? null : (selected?.path ?? null))} />
          <span>Follow newest</span>
        </label>
        <div class="segmented" role="group" aria-label="Shading">
          <button
            type="button"
            aria-pressed={!flat}
            onClick={() => {
              setFlat(false);
              savePrefs({ flat: false });
            }}
          >
            Smooth
          </button>
          <button
            type="button"
            aria-pressed={flat}
            onClick={() => {
              setFlat(true);
              savePrefs({ flat: true });
            }}
          >
            Flat
          </button>
        </div>
        <label class="inline-field">
          <span>Up</span>
          <select
            value={up}
            onChange={(event) => {
              const axis = event.currentTarget.value as UpAxis;
              setUp(axis);
              savePrefs({ up: axis });
            }}
            aria-label="Up direction of the model"
          >
            {UP_CHOICES.map((axis) => (
              <option key={axis} value={axis}>
                {axis.replace("-", "−")}
              </option>
            ))}
          </select>
        </label>
        <button type="button" class="button" onClick={() => viewer.current?.resetView()} disabled={shown === null}>
          <Icon name="fit" size={16} /> Reset view
        </button>
      </div>
      <p class="help viewer-note">The model has no physical scale and no fixed up direction; choose the axis that looks upright.</p>
    </section>
  );
}
