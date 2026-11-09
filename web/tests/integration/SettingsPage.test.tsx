import { describe, expect, it } from 'vitest'
import SettingsPage from '../../src/components/SettingsPage'
import { setup, screen } from '../utils/render'

describe('SettingsPage', () => {
  it('loads current settings, saves changes, and reports validation errors', async () => {
    const { user } = setup(<SettingsPage />)
    expect(await screen.findByText('/tmp/appdata')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Default greeting'), 'Howdy')
    await user.type(screen.getByLabelText('Data folder'), '/var/appdata')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))

    expect(await screen.findByText('Saved')).toBeInTheDocument()
    expect(await screen.findByText('Howdy')).toBeInTheDocument()

    await user.clear(screen.getByLabelText('Data folder'))
    await user.type(screen.getByLabelText('Data folder'), 'relative/path')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('the data folder must be an absolute path')
  })

  it('keeps existing values when saving with empty inputs', async () => {
    const { user } = setup(<SettingsPage />)
    await screen.findByText('/tmp/appdata')
    await user.click(screen.getByRole('button', { name: 'Save settings' }))
    expect(await screen.findByText('Saved')).toBeInTheDocument()
  })
})
