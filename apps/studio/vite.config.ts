import { cpSync, createReadStream, existsSync, statSync } from "node:fs";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = resolve(here, "../../tests/fixtures/dense-run-sphere");

const types: Record<string, string> = {
  ".png": "image/png",
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".stl": "model/stl",
};

/** Ships the recorded sphere run as `demo/`: served from the fixture in dev, copied into `dist` on build. */
function demoBundle(): Plugin {
  return {
    name: "crisp3ds-demo-bundle",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = (request.url ?? "").split("?")[0] ?? "";
        const at = path.indexOf("/demo/");
        if (at < 0) return next();
        const target = normalize(join(fixture, decodeURIComponent(path.slice(at + 6))));
        if (!target.startsWith(fixture + sep) || !existsSync(target) || !statSync(target).isFile()) return next();
        response.setHeader("Content-Type", types[extname(target)] ?? "application/octet-stream");
        response.setHeader("Content-Length", statSync(target).size);
        createReadStream(target).pipe(response);
      });
    },
    closeBundle() {
      if (!existsSync(fixture)) throw new Error("demo fixture missing: " + fixture);
      if (existsSync(resolve(here, "dist"))) cpSync(fixture, resolve(here, "dist/demo"), { recursive: true });
    },
  };
}

export default defineConfig({
  // Relative asset URLs: the build works from any base path (engine --static, GitHub Pages, Tauri).
  base: "./",
  clearScreen: false,
  plugins: [demoBundle()],
  server: { host: "127.0.0.1", port: 1430, strictPort: true },
  preview: { host: "127.0.0.1", port: 1431, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
  worker: { format: "es" },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
