import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes web/dist, which `dd serve` serves at http://<host>:8000.
// `npm run dev` runs a hot-reloading copy on http://localhost:5173 that
// forwards /ws and /api to a `dd serve` running on port 8000 - for working on
// the page itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/api": "http://localhost:8000",
    },
  },
});
