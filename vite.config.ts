import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
// `base` must match the GitHub Pages sub-path (https://<user>.github.io/openlow/)
// in production builds, while staying at root ('/') during local dev.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/openlow/' : '/',
  plugins: [react(), tailwindcss()],
}))
