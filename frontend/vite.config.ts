/// <reference types="vitest" />
// vite.config.ts — build tool and dev server configuration.
//
// The dev PROXY is the important part. The React app runs on :5173 and calls
// "/api/..." and "/health"; Vite forwards those to FastAPI on :8000. The
// browser therefore only ever talks to :5173, so there is no CORS problem in
// development, and the same relative URLs keep working in production behind a
// single host.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
  // Unit tests live next to the code they cover, as *.test.ts.
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
