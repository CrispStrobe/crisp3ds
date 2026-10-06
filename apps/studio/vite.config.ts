import { cpSync, createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
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

export default defineConfig({
  // Relative asset URLs: the build works from any base path (engine --static, GitHub Pages, Tauri).
  base: "./",
  clearScreen: false,
  define: { __BROWSER_ENGINE__: JSON.stringify(engineBuilt) },
  plugins: [
    // The recorded sphere run as `demo/`.
    mounted("crisp3ds-demo-bundle", "demo", fixture, null, true),
    ...(engineBuilt ? [mounted("crisp3ds-browser-engine", "engine", enginePackage, engineFiles, false)] : []),
  ],
  server: { host: "127.0.0.1", port: 1430, strictPort: true },
  preview: { host: "127.0.0.1", port: 1431, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
  worker: { format: "es" },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
