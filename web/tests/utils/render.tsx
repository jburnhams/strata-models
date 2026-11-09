import type { ReactElement } from 'react'
import userEvent from '@testing-library/user-event'
import { renderWithStrata } from 'strata-react/testing'

/** Render inside a router and a fresh query client (as the app runs, via strata-react/testing) and get a ready `user` (user-event) beside the usual RTL queries. Use instead of calling `render`/`fireEvent` directly. `route` sets the starting URL, e.g. `/library?q=dune`. */
export function setup(ui: ReactElement, options?: Parameters<typeof renderWithStrata>[1]) {
  return { user: userEvent.setup(), ...renderWithStrata(ui, options) }
}
export { default as userEvent } from '@testing-library/user-event'
export { screen, waitFor, within, act } from '@testing-library/react'
