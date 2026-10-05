# Crisp3DS Studio

The front end for Crisp3DS's photo-to-STL reconstruction. It shows a run as it
happens: the stages, the surface getting better step by step in a 3D view, the
diagnostic sheets, and the numbers.

Studio does not compute anything. A reconstruction runs on a desktop computer
(the *engine*); Studio is a client of one, or plays back a recorded run. It
relies only on [`docs/ENGINE-CONTRACT.md`](../../docs/ENGINE-CONTRACT.md).

It is one static web app. The same build is meant to be served by the engine,
hosted on any web server, and wrapped by Tauri 2 for desktop and mobile.

| Recorded run, desktop | Live engine run, phone |
| --- | --- |
| ![Run view of the demo recording](docs/replay-desktop.jpg) | ![Live run on a phone](docs/engine-live-phone.jpg) |

More screenshots are listed under [Verification](#verification).

## Running it

Node 22.18 or newer.

```sh
cd apps/studio
npm ci
npm run dev        # http://127.0.0.1:1430, demo served straight from tests/fixtures
npm run build      # typecheck, then static files in dist/
npm run preview    # serves dist/ on http://127.0.0.1:1431
npm run lint       # tsc --noEmit (strict)
npm test           # vitest
```

`dist/` uses relative URLs only and hash routes (`#/...`), so it works from any
base path without server rewrites: a site root, `https://user.github.io/repo/`,
the engine's `--static`, or a `tauri://` origin.

### Demo (no engine)

Open the app and choose **Play the demo**, or go to `#/replay`. The recording
is `tests/fixtures/dense-run-sphere/`, copied into `dist/demo/` at build time
(it is not duplicated in git).

### A recorded run (replay bundle)

Make a bundle with
`python -m scripts.turntable_mesh.export_replay --run <run> --output <bundle>`,
put it on any web server, and enter its address under **Recorded run**, or
open `#/replay?bundle=<address>`. The address may be relative to the app
(`big/`) or absolute. A bundle on another origin must be served with CORS
headers.

Playback follows the recorded timing. Speeds are 1×, 4×, 16× and instant (4× to
begin with; the choice is remembered). Pause, step one event forwards or
backwards, or drag the slider to any event; going backwards rebuilds the state
from the first event.

### A live engine

```sh
python -m scripts.turntable_mesh.engine_server --runs <dir> --data <dir> \
  --static apps/studio/dist --port 8765
```

Open `http://127.0.0.1:8765/`. Studio notices that an engine is serving the
page and offers its address; **Connect**, then start a run or open one. A
Studio served from elsewhere (the dev server, a phone on the same network)
connects the same way by typing the engine's address and, if the engine was
started with `--token`, the token.

The address and token are kept in this browser's `localStorage`. The token is
sent only in the `Authorization` header; it is never logged and never put in a
URL. With a token, images are fetched by script and shown through object URLs,
because an `<img>` cannot send the header.

## What it shows

- **Connection**: demo, a bundle address, or an engine (address and token).
- **Runs** (engine): status, the running stage with its progress, start time;
  refreshed every three seconds. **New run**: inputs folder, optional reference
  scan, device, name, and a settings form generated from `GET /api/settings`
  (grouped by `group`, `meaning` as help text, one typed input per `kind`,
  lists as comma-separated values). Only values that differ from the defaults
  are sent. "Reset to defaults" and a per-field "Default: ..." link undo
  changes. A `400` from the engine is shown at the field(s) its sentence names
  and once at the form.
- **Run view**:
  - Stage timeline with state, elapsed time, progress and the latest message;
    overall status; cancel (engine); errors with the engine's message.
  - **Surface**: the newest `preview_mesh` / `final_mesh` appears by itself. A
    step strip goes back and forth between surfaces; picking one turns "Follow
    newest" off. The camera stays put when surfaces are swapped and frames the
    bounding box for the first one. Orbit, zoom and pan with mouse, touch or
    keyboard (arrows, `+`, `-`, `0`). Smooth or flat shading. Triangle count.
    "Up" offers ±X, ±Y, ±Z, because the model's frame has no fixed up
    direction; no units are shown, because it has no physical scale.
  - **Diagnostics**: every sheet as a card in arrival order, grouped by stage,
    depth sheets ordered by `level`. A card opens a full-window viewer with
    zoom and pan (wheel, pinch, drag, double-click, `+` `-` `0` `1`, `[` `]`
    for the neighbouring sheets).
  - **Numbers**: `metric` events (latest value per name) and the `report`
    artifacts: mesh report, photo check, scanner evaluation (F1 per threshold,
    accuracy and completeness medians, and the `warnings` text, which is where
    the handedness warning arrives). Unknown JSON is shown as written. Every
    report has its full JSON in a collapsible block.
- Light and dark themes following the system, with a toggle. Layout from
  360 px upwards. All controls are reachable by keyboard and have visible
  focus; images carry their artifact label as text alternative. No external
  requests: system fonts, inline icons, everything bundled.

## Architecture

```
sources  ──events──▶  reducer  ──state──▶  views
(replay | http | local)  (pure)            (Preact components, three.js viewer)
```

```
src/
  core/        no DOM, no network; unit-tested
    events.ts      event type, log parsing (half-written last line is not consumed)
    reducer.ts     (state, event) -> state, plus selectors (sheet grouping, elapsed time, STL size)
    replay.ts      ReplayPlayer: recorded timing, speed, pause, scrub, step; clock injected
    settings.ts    settings form logic: text <-> typed values, diff against defaults, error placement
    reports.ts     report JSON -> rows; unknown shapes fall through to raw JSON
    stl.ts         binary STL -> indexed mesh (welded vertices, smooth normals, bounds), resumable
    format.ts      durations, sizes, counts
  sources/     no DOM; `fetch` and timers injected; unit-tested
    types.ts       RunSource and Engine: the one interface the views use
    replaySource.ts   bundle -> ReplayPlayer -> updates
    httpEngine.ts     HttpEngine (health, settings, runs, start) and HttpRunSource (polling, files, cancel)
    localEngine.ts    STUB for the Tauri shell: same shape, not implemented
    runStore.ts       folds a source's updates into RunState with the reducer
  viewer/      three.js; loaded on demand
    viewer.ts      scene, camera, controls, lighting, up axis, render on demand
    meshLoader.ts  download with progress, parse in a worker, LRU cache of parsed meshes
    stl.worker.ts  the worker
  ui/          Preact components and the stylesheet's class names
```

Things worth knowing:

- **One reducer.** Replay, HTTP and (later) local all deliver
  `{type: "events" | "reset" | "link"}` updates. `events` are appended;
  `reset` (replay scrubbing backwards) recomputes from the first event. Unknown
  event types, artifact kinds and fields are ignored; unknown stages are
  appended to the timeline; an unknown schema name gives a notice and the rest
  still works.
- **Polling.** `GET .../events?since=N` once a second. On an error the wait
  becomes 1, 2, 4, 8, then 15 s, the notice says why, and the cursor is kept,
  so nothing is lost or applied twice. `401`, `403` and `404` stop polling.
  Polling ends at `run_finished`.
- **Meshes.** Only the surface that is to be shown is downloaded. Parsing
  happens in a worker (in slices on the main thread if a worker cannot start).
  Identical vertices are welded, so a closed mesh takes about half of the
  STL's size in memory and smooth normals come for free; flat shading is done by the
  material on the same buffers. GPU buffers exist only for the mesh on screen
  and are disposed on every swap. Parsed meshes are cached up to 192 MB, least
  recently used first out.
- **The final mesh.** A binary STL is exactly `84 + 50 × triangles` bytes, so
  its size is known from the event. Above 12 MB (or when `triangles` is
  missing) the final mesh is not fetched until the "Load final surface (51 MB)"
  button is pressed; the newest preview stays on screen meanwhile.
- **Rendering on demand.** There is no animation loop; a still model costs
  nothing.

### Dependencies

| Package | Use | License |
| --- | --- | --- |
| `three` 0.186 | 3D viewer | MIT |
| `preact` 11 | view library (about 5 kB gzipped) | MIT |
| `vite` 8, `vitest` 5 (dev) | build, tests | MIT |
| `typescript` 7 (dev) | typecheck; `npm run lint` is `tsc --noEmit` | Apache-2.0 |
| `@types/three`, `@types/node` (dev) | types | MIT |
| `playwright` 1.63 (dev) | screenshot and performance scripts only | Apache-2.0 |

No UI kit, CSS framework or state library. `typescript-eslint` is not used
because it does not support TypeScript 7 yet.

## Verification

Unit tests (`npm test`, 104 tests): the reducer on the recorded sphere run and
on unknown events, kinds, fields, stages and schema; half-written last lines;
replay timing, speeds, pause, scrub and step with a fake clock; the STL parser
on all four fixture meshes (triangle counts as announced, closed and
consistently oriented, Euler characteristic 2, same vertex count as the mesh
report) and on broken input; settings parsing, diffing and error placement;
HTTP polling, backoff and cursor handling, token handling, and the replay
source, with a fake `fetch`.

In a real browser (headless Chromium through Playwright), against the built
app:

```sh
npx playwright install --only-shell chromium
npm run build && npm run preview &
npm run screenshots                                   # replay
STUDIO_URL=http://127.0.0.1:8765/ STUDIO_ENGINE_INPUTS=sphere STUDIO_ENGINE_EXTRAS=1 \
  npm run screenshots                                 # live engine, see the script's header
```

The engine pass connects through the form, provokes a validation error, starts
a small synthetic run through the settings form, watches it live on a desktop
and a phone viewport at the same time, waits for the end, then cancels a second
run. It was also run against an engine started with `--token` (wrong token
refused with a clear message; images and meshes load with the right one).

| | |
| --- | --- |
| ![](docs/replay-desktop.jpg) Recording, finished, 1440 px | ![](docs/replay-desktop-midway.jpg) Scrubbed back to event 12: stereo running, an earlier surface |
| ![](docs/replay-desktop-dark.jpg) Dark theme | ![](docs/lightbox-desktop.jpg) Sheet viewer, zoomed |
| ![](docs/engine-live-desktop.jpg) Live engine run, first surface | ![](docs/engine-done-desktop.jpg) The same run, finished |
| ![](docs/new-run-validation.jpg) The engine's validation message at its field | ![](docs/engine-cancelled-desktop.jpg) A run cancelled from the UI |
| ![](docs/runs-desktop.jpg) Runs list | ![](docs/connection-desktop.jpg) Connection |

| 390 px | 360 px | Gallery, 390 px | Live, 390 px |
| --- | --- | --- | --- |
| ![](docs/replay-phone.jpg) | ![](docs/replay-phone-360.jpg) | ![](docs/replay-phone-gallery.jpg) | ![](docs/engine-live-phone.jpg) |

### Large runs

No real large bundle was available, so the large case was synthesised:
the fixture's meshes repeated to about 250 000 triangles per preview and
1 014 000 triangles (50.7 MB) for the final STL, and the sheets scaled up to
4200 px wide. `scripts/perf-check.mjs` measures it. On an Apple M1 with the
real GPU (`STUDIO_GL=metal`), served from localhost:

- only the newest preview is downloaded; the final mesh waits for its button;
- first surface (252 000 triangles) on screen 0.4 to 1.2 s after opening the
  page;
- the 50.7 MB final mesh on screen 0.9 to 1.5 s after the click, longest
  main-thread pause 20 to 70 ms;
- swapping between cached surfaces 40 to 80 ms; 30 orbit steps on the final
  mesh in about 0.5 s;
- opening a 4200 × 4200 sheet at 1:1 pauses the page for 0.2 to 0.4 s (the
  browser decoding 17 megapixels).

These are six runs on a machine that was busy with other work (load average
15); two earlier runs showed pauses of 0.8 and 3 s while the page started,
which did not recur and were not traced to Studio's code. Before the download
loop was made to yield, the same final mesh froze the page for 0.7 s; that is
fixed and is the kind of thing the script is for.

A **real** megapixel run (Bunny or Dragon) has not been tested: real meshes
weld differently from repeated spheres, and real sheets may compress and decode
differently. Phones have not been measured at all. Run the script against one when it exists:

```sh
STUDIO_GL=metal STUDIO_BUNDLE=<address> node scripts/perf-check.mjs
```

## Implemented, and not

Implemented: everything under [What it shows](#what-it-shows); the replay and
HTTP sources; the demo.

Not implemented:

- **The local source** (`sources/localEngine.ts`) is a stub that rejects with
  "not implemented". Nothing in the UI offers it yet.
- **Photo upload and capture, camera recovery, segmentation**: not in the
  contract yet, so there is no screen for them. A run starts from a path on the
  engine's machine, typed by hand; there is no file browser because the engine
  has no endpoint to list its data directory.
- **Thumbnails.** Gallery cards show the full sheet scaled down. That is fine
  for megapixel sheets on a desktop and wasteful on a phone; the contract has
  no thumbnail artifact.
- **Deleting or renaming runs, downloading the STL through a button**: no
  endpoint for the first two; the STL is at
  `<engine>/api/runs/<id>/files/mesh/mesh.stl`.
- **Push updates.** Studio polls, as the contract says.
- **Mesh formats other than binary STL.** A text STL is refused with a clear
  message.
- **Installable PWA / offline shell**: no service worker.
- **Translations.** English only; all text is in the components.

One convention beyond the event log: the photo check writes
`check/result.json` (contract section 1) but no `report` event announces it, so
for a live engine Studio asks for that file once the check stage is done and
shows it if it is there. A replay bundle does not contain the file.

## The Tauri wrap

The intent is that wrapping needs no UI changes: the shell provides an engine
and Studio talks to it through `Engine` / `RunSource`.

**Desktop (macOS, Windows, Linux), first step.** Let the shell do what a
person does today: start `engine_server.py` as a sidecar on a free localhost
port with a random `--token`, and hand address and token to the web view. The
front end then uses `HttpEngine` unchanged. Concretely:

1. Add `src-tauri/` next to this app (or point `apps/desktop/src-tauri` at
   `apps/studio/dist` via `build.frontendDist`). `base: "./"` and hash routes
   already suit the `tauri://localhost` origin.
2. In Rust: pick a port, generate a token, spawn the Python sidecar
   (`tauri-plugin-shell`), wait for `GET /api/health`, kill the process tree on
   exit. Expose one command, e.g. `engine_endpoint() -> { url, token }`.
3. In `main.tsx`: if `window.__TAURI__` exists, call that command and open
   `#/engine` with an `HttpEngine` built from it instead of showing the
   connection form. That is the only front-end change, about ten lines.
4. CSP: allow `connect-src` and `img-src` for `http://127.0.0.1:*` and `blob:`,
   and `worker-src 'self' blob:`.
5. Add a folder picker (`tauri-plugin-dialog`) for the inputs path. The engine
   only accepts paths under `--data`, so either start it with `--data` set to
   the chosen folder's parent or extend the contract.

**Desktop, later.** Fill in `LocalEngine` / `LocalRunSource` to drop the HTTP
hop: the shell tails `events.jsonl` (remembering the line count, ignoring a
last line without a newline, exactly as `core/events.ts` does) and emits
complete lines on a Tauri channel; files are read through the asset protocol
(`convertFileSrc`) for images and as bytes for STL and JSON; cancelling creates
`<run>/cancel`. The interface is in `sources/types.ts` and the HTTP source is
the model to follow. This only pays off if the HTTP server becomes a burden.

**Mobile (iOS, Android).** Phones do not compute. The Tauri mobile shell is
the same web app with the connection screen: demo, recorded bundles and a
remote engine. What a phone build needs beyond `tauri ios init` /
`tauri android init`:

- Cleartext HTTP to a LAN engine is blocked by default: an App Transport
  Security exception on iOS (`NSAllowsLocalNetworking`, plus the local-network
  usage description) and `usesCleartextTraffic` or a network security config on
  Android. Better: put the engine behind TLS.
- Engine discovery (mDNS, or a QR code shown by the desktop app carrying
  address and token) so nobody types an IP address on a phone. The token would
  then move from `localStorage` to the platform keychain.
- Safe-area insets are already handled in the CSS (`env(safe-area-inset-*)`,
  `viewport-fit=cover`); touch targets are 38 px or more.
- Photo capture and upload wait for `POST /api/uploads` in the contract.
- Memory: a 50 MB final STL becomes about 24 MB of buffers, plus the 50 MB
  download while it is being parsed. That should be tested on a real phone before enabling the
  "load final surface" button there without a warning; a decimated preview
  artifact (the contract mentions GLB) would be the better answer.
