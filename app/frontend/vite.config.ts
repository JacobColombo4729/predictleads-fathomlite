import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is proxied to the demo backend so the session cookie stays same-origin with the page.
const backend = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } };

export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy: backend },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy: backend },
});
