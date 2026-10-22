import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` on :5173 talks to the API on :8765; `npm run build` writes dist/, which the API serves itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
