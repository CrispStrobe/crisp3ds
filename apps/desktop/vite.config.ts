import { defineConfig } from "vitest/config";

export default defineConfig({
  clearScreen: false,
  server: { host: "127.0.0.1", port: 1420, strictPort: true },
  test: { include: ["src/**/*.test.ts"] },
});
