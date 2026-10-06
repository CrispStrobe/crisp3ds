// A run of the dense pipeline in the browser, from a map of files.
//
//   import { createRun, settingsSchema } from "./crisp3ds-dense.js";
//   const run = await createRun({
//     files,                         // Map or object: path inside the inputs directory -> Uint8Array
//     options: { settings: { grid: 96 } },   // fields of RunOptions; output and inputs are filled in
//     onEvent: (event) => { ... },   // the objects of events.jsonl, as they are emitted
//   });
//   const report = await run.finished;      // content of pipeline.json; rejects on failure or cancel
//   const stl = run.file("mesh/mesh.stl");  // Uint8Array of any file of the run directory
//   run.cancel();                           // optional, at any time
//   run.dispose();                          // frees the run's files
//
// The engine works on the thread that calls it and holds it during the CPU
// passes; use it from a worker (see worker.js) to keep a page responsive.

import init, * as engine from "./pkg/crisp3ds_dense_web.js";

let ready;
let wasm;
let counter = 0;
const sources = new Map();

function lookup(path, probe) {
  for (const [prefix, inputs] of sources) {
    if (path.startsWith(prefix)) {
      const bytes = inputs.get(path.slice(prefix.length));
      return probe ? bytes !== undefined : bytes;
    }
  }
  return undefined;
}

/** Loads the WebAssembly module once. */
export async function load() {
  ready ??= init().then((exports) => {
    wasm = exports;
  });
  await ready;
}

/** Name, group, meaning, kind and default of every setting. */
export async function settingsSchema() {
  await load();
  return JSON.parse(engine.settingsSchema()).settings;
}

/** The default settings by name. */
export async function defaultSettings() {
  await load();
  return JSON.parse(engine.defaultSettings());
}

/** Bytes of the WebAssembly memory (it grows and never shrinks) and of the files held in it. */
export function memory() {
  return { wasm: wasm ? wasm.memory.buffer.byteLength : 0, files: wasm ? engine.storedBytes() : 0 };
}

export async function createRun({ files, options = {}, onEvent } = {}) {
  await load();
  const root = `run-${++counter}`;
  // The input files stay here, in JavaScript memory; the engine asks for one at a time.
  const inputs = files instanceof Map ? files : new Map(Object.entries(files ?? {}));
  sources.set(`${root}/inputs/`, inputs);
  engine.setFileSource(lookup);
  const handle = new engine.Run();
  const text = JSON.stringify({ ...options, output: `${root}/run`, inputs: `${root}/inputs` });
  const finished = handle.start(text, onEvent ?? null).then((report) => JSON.parse(report));
  return {
    finished,
    cancel: () => handle.cancel(),
    /** A file of the run directory, e.g. the `path` of an artifact event; undefined if it does not exist. */
    file: (path) => engine.getFile(`${root}/run/${path}`),
    /** [[path, bytes], ...] of everything the run wrote. */
    files: () => JSON.parse(engine.listFiles(`${root}/run`)).map(([path, size]) => [path.slice(`${root}/run/`.length), size]),
    dispose: () => {
      sources.delete(`${root}/inputs/`);
      engine.removeTree(root);
      handle.free();
    },
  };
}
