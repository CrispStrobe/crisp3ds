import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  downloadExample,
  EXAMPLES_MANIFEST,
  EXAMPLES_PURPOSE,
  parseExampleManifest,
  runAttribution,
  type DownloadProgress,
  type ExampleManifest,
  type ExampleObject,
} from "../core/examples";
import { formatBytes } from "../core/format";
import type { BrowserEngine } from "../sources/browserEngine";
import { AppExampleStore, BrowserExampleStore } from "../sources/exampleStores";
import type { Bridge } from "../sources/localEngine";
import { describe } from "../sources/transport";
import type { Engine } from "../sources/types";
import { navigate } from "./app";

interface Props {
  engine: Engine;
  /** The shell's command bridge, when the engine is the one built into the app. */
  bridge: Bridge | null;
  /** Where the list comes from; the dataset repository by default. */
  manifestUrl?: string;
}

type Row = { state: "unknown" | "absent" | "partial" | "ready"; have: number } | { state: "downloading"; progress: DownloadProgress } | { state: "failed"; message: string };

/**
 * Real photo sets that can be downloaded on request and reconstructed: nothing of them comes
 * with the app. Before anything is fetched the source, the license and the download size are
 * shown; the attribution is kept with every run made from them.
 */
export function Examples({ engine, bridge, manifestUrl = EXAMPLES_MANIFEST }: Props) {
  const [manifest, setManifest] = useState<ExampleManifest | null>(null);
  const [problem, setProblem] = useState("");
  const [rows, setRows] = useState<Record<string, Row>>({});
  const [confirm, setConfirm] = useState<ExampleObject | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(manifestUrl)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`The list of example objects could not be loaded (${response.status}).`))))
      .then((json) => alive && setManifest(parseExampleManifest(json, manifestUrl)))
      .catch((reason) => alive && setProblem(describe(reason)));
    return () => {
      alive = false;
      abort.current?.abort();
    };
  }, [manifestUrl]);

  const store = useMemo(() => {
    if (manifest === null) return null;
    if (engine.kind === "browser") return new BrowserExampleStore(manifest, engine as BrowserEngine);
    if (engine.kind === "local" && bridge !== null) return new AppExampleStore(bridge, engine);
    return null;
  }, [manifest, engine, bridge]);

  const refresh = async (object: ExampleObject) => {
    if (store === null) return;
    const have = (await store.have(object)).size;
    const total = object.files.length + 1;
    setRows((current) => ({ ...current, [object.id]: { state: have === 0 ? "absent" : have === total ? "ready" : "partial", have } }));
  };
  useEffect(() => {
    for (const object of manifest?.objects ?? []) void refresh(object);
  }, [store]);

  const run = async (object: ExampleObject) => {
    if (manifest === null || store === null) return;
    setConfirm(null);
    const controller = new AbortController();
    abort.current = controller;
    try {
      await downloadExample(manifest, object, store, {
        signal: controller.signal,
        onProgress: (progress) => setRows((current) => ({ ...current, [object.id]: { state: "downloading", progress } })),
      });
      await refresh(object);
      const id = await store.start(manifest, object, runAttribution(manifest, object));
      navigate(`#/engine/run/${encodeURIComponent(id)}`);
    } catch (reason) {
      if (controller.signal.aborted) await refresh(object);
      else setRows((current) => ({ ...current, [object.id]: { state: "failed", message: describe(reason) } }));
    }
  };

  const remove = async (object: ExampleObject) => {
    if (store === null) return;
    try {
      await store.remove(object);
    } catch (reason) {
      setRows((current) => ({ ...current, [object.id]: { state: "failed", message: describe(reason) } }));
      return;
    }
    await refresh(object);
  };

  return (
    <section class="page narrow examples">
      <a class="back" href="#/">
        Connection
      </a>
      <h1>Example objects</h1>
      <p class="sub">
        Real turntable photo sets, one turn of 73 photos each, that can be downloaded and reconstructed here. They do not come
        with {engine.kind === "browser" ? "this page" : "the app"}: they are fetched on request from a public dataset repository,
        each file checked against its published checksum, and kept {engine.kind === "browser" ? "in this browser" : "in the app's data folder"} until you delete them.
      </p>
      {problem !== "" && <p class="notice bad">{problem}</p>}
      {manifest === null && problem === "" && <p role="status">Loading the list...</p>}
      {manifest !== null && (
        <>
          <section class="card attribution-card" aria-labelledby="ex-source">
            <h2 id="ex-source">Source and license</h2>
            <p>{manifest.attribution.text}</p>
            <p class="help">{EXAMPLES_PURPOSE}</p>
            <p class="help">
              {manifest.attribution.source !== undefined && (
                <>
                  Original dataset: <span class="mono wrap">{manifest.attribution.source}</span>.{" "}
                </>
              )}
              Downloaded from: <span class="mono wrap">{manifest.baseUrl}</span>
            </p>
          </section>
          {store === null && <p class="notice">Example objects are reconstructed by the engine built into the app or by the engine in this browser.</p>}
          <ul class="example-list">
            {manifest.objects.map((object) => {
              const row = rows[object.id] ?? { state: "unknown", have: 0 };
              const busy = Object.values(rows).some((other) => other.state === "downloading");
              return (
                <li key={object.id} class="example">
                  <div class="example-text">
                    <strong>{object.name}</strong>
                    <span class="help">
                      {object.photos} photos · {formatBytes(object.bytes)}
                      {row.state === "ready" && " · downloaded"}
                      {row.state === "partial" && ` · ${row.have} of ${object.files.length + 1} files downloaded`}
                    </span>
                    {row.state === "downloading" && (
                      <span class="help" role="status">
                        Downloading {row.progress.done} of {row.progress.total} files, {formatBytes(row.progress.bytes)} of {formatBytes(row.progress.totalBytes)}
                        <progress max={row.progress.totalBytes} value={row.progress.bytes} />
                      </span>
                    )}
                    {row.state === "failed" && <span class="field-error">{row.message}</span>}
                  </div>
                  <div class="actions">
                    {row.state === "downloading" ? (
                      <button type="button" class="button" onClick={() => abort.current?.abort()}>
                        Cancel
                      </button>
                    ) : (
                      <>
                        <button
                          type="button"
                          class="button primary"
                          disabled={store === null || busy}
                          onClick={() => (row.state === "ready" ? void run(object) : setConfirm(object))}
                        >
                          {row.state === "ready" ? "Reconstruct" : "Download and reconstruct"}
                        </button>
                        {(row.state === "ready" || row.state === "partial") && (
                          <button type="button" class="button" disabled={busy} onClick={() => void remove(object)}>
                            Delete download
                          </button>
                        )}
                      </>
                    )}
                  </div>
                  {confirm?.id === object.id && (
                    <div class="example-confirm" role="dialog" aria-label={`Download ${object.name}`}>
                      <p>
                        Download {object.photos} photos of the {object.name} ({formatBytes(object.bytes)}) from <span class="mono wrap">{manifest.baseUrl}</span>, then
                        reconstruct them with threshold masks and turntable cameras? The source and license above are kept with the run.
                      </p>
                      <div class="actions">
                        <button type="button" class="button primary" onClick={() => void run(object)}>
                          Download and reconstruct
                        </button>
                        <button type="button" class="button" onClick={() => setConfirm(null)}>
                          Not now
                        </button>
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </section>
  );
}
