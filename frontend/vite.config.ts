import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite is the build tool. It does two jobs:
//   - during development it serves the app instantly with hot reloading
//   - for production it compiles everything into plain HTML/CSS/JS
//
// We send that compiled output into ../static, which is exactly where the
// FastAPI backend looks for it. One build step, no copying files around.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
    // Split the big libraries into their own files so the browser can cache
    // them separately from our own code, which changes far more often.
    rollupOptions: {
      output: {
        manualChunks: {
          deck: ["@deck.gl/core", "@deck.gl/layers", "@deck.gl/react"],
          charts: ["recharts"],
        },
      },
    },
  },
  server: {
    port: 5173,
    // While developing, anything starting with /api is forwarded to the Python
    // backend on port 7860. This means the frontend code can just call
    // "/api/predict" and it works identically in development and production.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7860",
        changeOrigin: true,
      },
    },
  },
});
