import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// /api is proxied to the demo backend so the session cookie stays same-origin with the page.
// PREDICTLEADS_FRONTEND_PORT and PREDICTLEADS_BACKEND_PORT move a second copy next to the demo (run_jacob.ps1 / .sh);
// unset, the ports are 5173 and 8000 as always.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "PREDICTLEADS_");
  const backend = { "/api": { target: `http://127.0.0.1:${env.PREDICTLEADS_BACKEND_PORT || "8000"}`, changeOrigin: false } };
  return {
    plugins: [react()],
    server: { host: "127.0.0.1", port: Number(env.PREDICTLEADS_FRONTEND_PORT || 5173), strictPort: true, proxy: backend },
    preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy: backend },
  };
});
