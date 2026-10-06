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
// From turntable photos instead of an inputs directory:
//   const run = await createRun({ photos, calibration, onEvent });   // photos: name -> Uint8Array
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

// Folders the host supplies: the names directly inside a source prefix (photos only; an inputs
// directory is read by the paths in cameras.json and needs no listing).
const listed = new Set();

function list(directory) {
  const folder = `${directory}/`;
  for (const prefix of listed) {
    if (!folder.startsWith(prefix)) continue;
    const inside = folder.slice(prefix.length);
    const names = new Set();
    for (const key of sources.get(prefix).keys()) if (key.startsWith(inside)) names.add(key.slice(inside.length).split("/")[0]);
    return names.size ? [...names] : undefined;
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

/**
 * How a run may start in this browser: `{schema, platform, start_points: [...]}`, each start point
 * with its fields, stages and the providers that run here (with their options), for a form built
 * from data. The photos start point maps to `createRun({ photos, calibration, options })`, with the
 * chosen providers' words in `options.photo_options`.
 */
export async function describe() {
  await load();
  return JSON.parse(engine.describe());
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

/**
 * A run from an inputs directory (`files`) or from turntable photos (`photos` and `calibration`).
 *
 *   photos:      Map or object: file name -> Uint8Array (PNG or JPEG), one turn in the order of the names
 *   calibration: the lens calibration (crisp3ds_lens_calibration_v1) as an object, JSON text or bytes
 *
 * From photos the masks and cameras are recovered first with the providers that need no other
 * program (threshold masks and the turntable solver by default); `options.photo_options` takes
 * further words of `crisp3ds-dense photos`. The photos stay in JavaScript memory (the map is
 * kept until `dispose`) and are handed to the engine one at a time.
 */
export async function createRun({ files, photos, calibration, options = {}, onEvent } = {}) {
  await load();
  const root = `run-${++counter}`;
  if (photos) return startFromPhotos(root, photos, calibration, options, onEvent);
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

function startFromPhotos(root, photos, calibration, options, onEvent) {
  if (calibration === undefined || calibration === null) throw new Error("a run from photos needs the lens calibration");
  const entries = photos instanceof Map ? [...photos] : Object.entries(photos);
  if (!entries.length) throw new Error("no photos");
  for (const [name] of entries) {
    if (name.includes("/")) throw new Error(`photo names are file names, not paths: ${name}`);
  }
  // The photos stay here, in JavaScript memory; the engine lists the folder and asks for one photo at a time.
  const prefix = `${root}/photos/`;
  sources.set(prefix, new Map(entries));
  listed.add(prefix);
  engine.setFileSource(lookup);
  engine.setFileLister(list);
  const lens = calibration instanceof Uint8Array ? calibration
    : new TextEncoder().encode(typeof calibration === "string" ? calibration : JSON.stringify(calibration));
  engine.putFile(`${root}/calibration.json`, lens);
  const handle = new engine.Run();
  const text = JSON.stringify({
    ...options,
    output: `${root}/run`,
    photos: `${root}/photos`,
    photo_options: ["--calibration", `${root}/calibration.json`, ...(options.photo_options ?? [])],
  });
  // Once the photos stage has written the scene, its files leave WebAssembly memory for this
  // side, like the inputs of a run from an inputs directory: the dense stages then start with
  // the engine's memory nearly empty instead of holding the scene behind their own buffers.
  const scene = `${root}/run/frontend/inputs`;
  const relay = (event) => {
    if (event.type === "stage_finished" && event.stage === "cameras") {
      const files = new Map();
      for (const [path] of JSON.parse(engine.listFiles(scene))) files.set(path.slice(scene.length + 1), engine.getFile(path));
      engine.removeTree(scene);
      sources.set(`${scene}/`, files);
      listed.add(`${scene}/`);
    }
    onEvent?.(event);
  };
  const finished = handle.start(text, relay).then((report) => JSON.parse(report));
  return {
    finished,
    cancel: () => handle.cancel(),
    file: (path) => engine.getFile(`${root}/run/${path}`) ?? (path.startsWith("frontend/inputs/") ? lookup(`${root}/run/${path}`, false) : undefined),
    files: () => JSON.parse(engine.listFiles(`${root}/run`)).map(([path, size]) => [path.slice(`${root}/run/`.length), size]),
    dispose: () => {
      for (const folder of [prefix, `${scene}/`]) {
        sources.delete(folder);
        listed.delete(folder);
      }
      engine.removeTree(root);
      handle.free();
    },
  };
}
