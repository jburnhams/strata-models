import { mergeConfig, defineConfig } from 'vitest/config'
import viteConfig from './vite.config'

// Unit tests run in node (pure logic); integration tests render components in jsdom with the backend mocked by msw.
export default mergeConfig(viteConfig, defineConfig({
  test: {
    projects: [
      { extends: true, test: { name: 'unit', include: ['tests/unit/**/*.test.{ts,tsx}'], environment: 'node' } },
      { extends: true, test: { name: 'integration', include: ['tests/integration/**/*.test.{ts,tsx}'], environment: 'jsdom', setupFiles: ['tests/utils/setup.ts'] } },
    ],
    coverage: { provider: 'v8', include: ['src/**/*.{ts,tsx}'], exclude: ['src/main.tsx'], reporter: ['text', 'lcov'] },
  },
}))
