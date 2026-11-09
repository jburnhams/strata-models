import type { ReactElement } from 'react'
import { render } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

/** Render and get a ready `user` (user-event) beside the usual RTL queries. Use instead of calling `render`/`fireEvent` directly. */
export function setup(ui: ReactElement, options?: Parameters<typeof render>[1]) {
  return { user: userEvent.setup(), ...render(ui, options) }
}
export { default as userEvent } from '@testing-library/user-event'
export { screen, waitFor, within, act } from '@testing-library/react'
