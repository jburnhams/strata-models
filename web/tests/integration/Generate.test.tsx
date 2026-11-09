import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { paramsOf } from '../../src/components/GeneratePage'
import { mock } from '../utils/server'
import { screen, setup, waitFor } from '../utils/render'

describe('Generate', () => {
  it('needs only a prompt, keeps the settings under Advanced, and shows the picture', async () => {
    const { user } = setup(<App />, { route: '/generate' })
    expect(await screen.findByText('No pictures yet')).toBeInTheDocument()
    const go = screen.getByRole('button', { name: 'Generate' }); expect(go).toBeDisabled()
    expect(screen.queryByLabelText('Seed')).not.toBeInTheDocument()
    await user.type(screen.getByLabelText('What should the picture show?'), 'a hare')
    await user.click(go)
    expect(await screen.findByRole('img', { name: 'a hare' })).toBeInTheDocument()
    expect(screen.getByText(/seed 5/)).toBeInTheDocument()
    expect(mock.calls).toEqual(['generate a hare'])
  })
  it('opens Advanced for model, size, seed and steps, and adds a reference picture', async () => {
    const { user } = setup(<App />, { route: '/generate' })
    await user.click(await screen.findByRole('button', { name: 'Advanced' }))
    expect(await screen.findByRole('option', { name: 'Default (flux3-klein)' })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Size and shape'), '832x1216')
    await user.type(screen.getByLabelText('Seed'), '42')
    const file = new File(['png'], 'ref.png', { type: 'image/png' })
    await user.upload(document.querySelector('input[type=file]') as HTMLInputElement, file)
    expect(await screen.findByText('1 reference picture added')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Clear references' }))
    expect(screen.getByText('Add a reference picture')).toBeInTheDocument()
  })
  it('says why when it fails, and lists earlier pictures', async () => {
    mock.images = [{ id: 1, app: 'web', model: 'flux3-klein', prompt: 'an owl', finished_at: 1, path: '/d/g/o.png', width: 1024, height: 1024, seed: 1, steps: 16, seconds: 9, peak_vram_mb: 6000 }]
    const { user } = setup(<App />, { route: '/generate' })
    expect(await screen.findByText('an owl')).toBeInTheDocument()
    await user.type(screen.getByLabelText('What should the picture show?'), 'fail')
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('stable-diffusion.cpp is not installed'))
  })
  it('sends only the settings that were set', () => {
    expect(paramsOf({ prompt: ' a ', size: '1216x832', seed: '', steps: '', guidance: '', refs: [] })).toEqual({ prompt: 'a', width: 1216, height: 832 })
    expect(paramsOf({ prompt: 'a', size: '768x768', seed: '3', steps: '8', guidance: '1.5', refs: ['/r.png'] })).toEqual({ prompt: 'a', width: 768, height: 768, seed: 3, steps: 8, cfg_scale: 1.5, ref_images: ['/r.png'] })
  })
})

import { http, HttpResponse, server } from '../utils/server'
import { imagesApi, evalsApi } from '../../src/api'
describe('API errors', () => {
  it('turns refusals into sentences', async () => {
    server.use(http.post('/api/images/upload', () => HttpResponse.json({ detail: 'a reference is a .png' }, { status: 400 })), http.post('/api/images/upload?x', () => HttpResponse.text('x')))
    await expect(imagesApi.upload(new File(['x'], 'a.exe'))).rejects.toThrow('a reference is a .png')
    server.use(http.post('/api/images/upload', () => new HttpResponse('boom', { status: 500, statusText: 'Server Error' })))
    await expect(imagesApi.upload(new File(['x'], 'a.png'))).rejects.toThrow('500 Server Error')
    server.use(http.get('/api/evals/:name/report', () => new HttpResponse('no', { status: 404, statusText: 'Not Found' })))
    await expect(evalsApi.report('x')).rejects.toThrow('404 Not Found')
    server.use(http.post('/api/evals/:name/run', () => new HttpResponse('x', { status: 409, statusText: 'Conflict' })))
    await expect(evalsApi.run('x')).rejects.toThrow('409 Conflict')
    expect(evalsApi.mediaUrl('a b', 'c')).toBe('/api/evals/a%20b/media/c')
    expect((await imagesApi.list(5)).items).toEqual([])
  })
  it('stops waiting when a job is withdrawn', async () => {
    server.use(http.get('/api/jobs/99', () => HttpResponse.json({ id: 99, status: 'cancelled', error: null })))
    await expect(imagesApi.generate(undefined, { prompt: 'x' })).rejects.toThrow('the job was cancelled')
  })
})
