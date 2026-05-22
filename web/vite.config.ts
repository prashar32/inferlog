import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In Docker the web container's nginx proxies /api/* to the services.
// For local `npm run dev`, mirror that here so the frontend code can use
// the same relative URLs in both cases. Ports match docker-compose.yml.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api/gateway": {
        target: "http://localhost:8086",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/gateway/, ""),
      },
      "/api/ingestion": {
        target: "http://localhost:8081",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/ingestion/, ""),
      },
    },
  },
});
