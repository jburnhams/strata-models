import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run build` writes into the Python package, so `./strata-models serve` serves the built app.
// `npm run dev` runs the dev server with hot reload and proxies /api to a running `./strata-models serve` (port 8366).
export default defineConfig({
  base: './',       // relative asset paths, so the app works at the server's root and behind a reverse proxy at a path (api.ts BASE does the same for the API)
  plugins: [react(), tailwindcss()],
  build: { outDir: '../src/stratamodels/server/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8366' } },
})
