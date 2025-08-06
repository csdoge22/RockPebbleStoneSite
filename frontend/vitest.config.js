// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom', // for DOM/react testing
    globals: true,         // use global test functions like `describe`, `it`
  },
})