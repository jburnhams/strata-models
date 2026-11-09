import { mergeConfig, defineConfig } from 'vitest/config'
import { readFileSync } from 'node:fs'
import baseConfig from './vitest.config'

// The coverage gate: unit + integration together must stay at or above the floors in coverage-floor.json.
// The floors only ratchet up: after CI is green, raise them to the floor of the new totals. Never lower them.
const floor = JSON.parse(readFileSync(new URL('./coverage-floor.json', import.meta.url), 'utf8'))

export default mergeConfig(baseConfig, defineConfig({
  test: { coverage: { thresholds: floor } },
}))
