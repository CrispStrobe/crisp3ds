import { cpSync, createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = resolve(here, "../../tests/fixtures/dense-run-sphere");
// The engine for "This browser": the WebAssembly package of crates/dense/web, built by
// crates/dense/web/build.sh into pkg/. When it has not been built, the app still builds
// and that mode says it is not included.
const enginePackage = resolve(here, "../../crates/dense/web");
const engineFiles = ["crisp3ds-dense.js", "pkg/crisp3ds_dense_web.js", "pkg/crisp3ds_dense_web_bg.wasm"];
const engineBuilt = engineFiles.every((name) => existsSync(join(enginePackage, name)));

const types: Record<string, string> = {
  ".png": "image/png",
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".stl": "model/stl",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
};

/** Serves files of `folder` under `/<mount>/` in dev, and copies them (or `only`) into `dist/<mount>` on build. */
function mounted(name: string, mount: string, folder: string, only: string[] | null, required: boolean): Plugin {
  return {
    name,
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = (request.url ?? "").split("?")[0] ?? "";
        const at = path.indexOf(`/${mount}/`);
        if (at < 0) return next();
        const relative = decodeURIComponent(path.slice(at + mount.length + 2));
        const target = normalize(join(folder, relative));
        if (!target.startsWith(folder + sep) || (only !== null && !only.includes(relative)) || !existsSync(target) || !statSync(target).isFile()) return next();
        response.setHeader("Content-Type", types[extname(target)] ?? "application/octet-stream");
        response.setHeader("Content-Length", statSync(target).size);
        createReadStream(target).pipe(response);
      });
    },
    closeBundle() {
      const out = resolve(here, "dist");
      if (!existsSync(out)) return;
      if (only === null) {
        if (!existsSync(folder)) {
          if (required) throw new Error(`${name}: missing ${folder}`);
          return;
        }
        cpSync(folder, join(out, mount), { recursive: true });
        return;
      }
      for (const file of only) {
        mkdirSync(dirname(join(out, mount, file)), { recursive: true });
        cpSync(join(folder, file), join(out, mount, file));
      }
    },
  };
}

// Lens calibrations that come with the app, offered by the photos start in "This browser".
const calibrations = resolve(here, "../../scripts/turntable_mesh/calibrations");
function calibrationIndex(): string {
  const rows = existsSync(calibrations)
    ? readdirSync(calibrations)
        .filter((name) => name.endsWith(".json"))
        .sort()
        .filter((name) => (JSON.parse(readFileSync(join(calibrations, name), "utf8")) as { schema?: string }).schema === "crisp3ds_lens_calibration_v1")
        .map((name) => ({ file: name, label: name.replace(/\.json$/, "") }))
    : [];
  return JSON.stringify(rows, null, 2) + "\n";
}
const calibrationsPlugin: Plugin = {
  name: "crisp3ds-calibrations",
  configureServer(server) {
    server.middlewares.use((request, response, next) => {
      const path = (request.url ?? "").split("?")[0] ?? "";
      const at = path.indexOf("/calibrations/");
      if (at < 0) return next();
      const name = decodeURIComponent(path.slice(at + "/calibrations/".length));
      if (name === "index.json") {
        response.setHeader("Content-Type", "application/json");
        return response.end(calibrationIndex());
      }
      const target = join(calibrations, name);
      if (name.includes("/") || !name.endsWith(".json") || !existsSync(target)) return next();
      response.setHeader("Content-Type", "application/json");
      createReadStream(target).pipe(response);
    });
  },
  closeBundle() {
    const out = resolve(here, "dist/calibrations");
    if (!existsSync(resolve(here, "dist"))) return;
    mkdirSync(out, { recursive: true });
    for (const row of JSON.parse(calibrationIndex()) as { file: string }[]) cpSync(join(calibrations, row.file), join(out, row.file));
    writeFileSync(join(out, "index.json"), calibrationIndex());
  },
};

export default defineConfig({
  // Relative asset URLs: the build works from any base path (engine --static, GitHub Pages, Tauri).
  base: "./",
  clearScreen: false,
  define: { __BROWSER_ENGINE__: JSON.stringify(engineBuilt) },
  plugins: [
    // The recorded sphere run as `demo/`.
    mounted("crisp3ds-demo-bundle", "demo", fixture, null, true),
    calibrationsPlugin,
    ...(engineBuilt ? [mounted("crisp3ds-browser-engine", "engine", enginePackage, engineFiles, false)] : []),
  ],
  server: { host: "127.0.0.1", port: 1430, strictPort: true },
  preview: { host: "127.0.0.1", port: 1431, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
  worker: { format: "es" },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
