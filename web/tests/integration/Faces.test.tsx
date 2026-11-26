import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { faceMock } from '../utils/faces'
import { screen, setup, waitFor, within } from '../utils/render'

// A removal waits out its Undo window (8 s); leaving the screen commits it, which is how these tests see the request without waiting.
const commits = async (unmount: () => void, call: string) => { unmount(); await waitFor(() => expect(faceMock.calls).toContain(call)) }

describe('Faces: people', () => {
  it('lists everyone named, most faces first, and finds a person by name', async () => {
    const { user } = setup(<App />, { route: '/faces' })
    expect(await screen.findByText(/3\speople named, with 6\sfaces confirmed\./)).toBeInTheDocument()
    const table = await screen.findByRole('grid', { name: 'People' })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map(r => within(r).getByRole('link').textContent)).toEqual(['Ada', 'Cy', 'Person 2'])
    expect(within(rows[0]).getByText('bright-horizons')).toBeInTheDocument()
    expect(within(rows[2]).getByText('–')).toBeInTheDocument()
    await user.type(screen.getByRole('searchbox', { name: 'Find a person' }), 'cy')
    await waitFor(() => expect(within(screen.getByRole('grid', { name: 'People' })).getAllByRole('row')).toHaveLength(2))
    await user.clear(screen.getByRole('searchbox', { name: 'Find a person' }))
    await user.type(screen.getByRole('searchbox', { name: 'Find a person' }), 'zed')
    expect(await screen.findByText('Nobody called “zed”')).toBeInTheDocument()
  })
  it('teaches the next step when nobody is named yet', async () => {
    faceMock.people = []
    setup(<App />, { route: '/faces' })
    expect(await screen.findByText('Nobody is named yet')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Unnamed groups' })).toHaveAttribute('href', '/faces?tab=groups')
  })
})

describe('Faces: a person', () => {
  it('shows their faces, opens one zoomed in its photo with the names of who is beside them', async () => {
    const { user } = setup(<App />, { route: '/faces/people/1' })
    expect(await screen.findByRole('heading', { name: 'Ada', level: 1 })).toBeInTheDocument()
    expect(screen.getByText(/3\sfaces confirmed, named in bright\-horizons\./)).toBeInTheDocument()
    const grid = await screen.findByRole('grid', { name: 'Faces of Ada' })
    await user.click(within(within(grid).getByRole('row', { name: 'Face 11' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Show in its photo' }))
    expect(await screen.findByRole('heading', { name: 'In its photo' })).toBeInTheDocument()
    const list = await screen.findByRole('listbox', { name: 'Faces in this photo' })
    expect(within(list).getAllByRole('option').map(o => o.textContent)).toEqual(['Ada', 'Not named yet']) // the "not a face" box is not drawn
    expect(within(list).getByRole('option', { name: 'Ada' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByLabelText(/Photo from bright-horizons, /)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Close the photo' }))
    expect(screen.queryByRole('heading', { name: 'In its photo' })).not.toBeInTheDocument()
    await user.dblClick(within(grid).getByRole('row', { name: 'Face 13' })) // opening a tile shows it in its photo too
    expect(await screen.findByText('No copy of this photo is kept')).toBeInTheDocument()
  })
  it('says when a photo has no kept copy, and when its size is not known', async () => {
    setup(<App />, { route: '/faces/people/1?face=13' })
    expect(await screen.findByText('No copy of this photo is kept')).toBeInTheDocument()
    expect(within(await screen.findByRole('listbox', { name: 'Faces in this photo' })).getByRole('option', { name: 'Cy' })).toBeInTheDocument()
  })
  it('says when a photo cannot be placed because its size is not known', async () => {
    setup(<App />, { route: '/faces/people/1?face=12' })
    expect(await screen.findByText("This photo's size is not known")).toBeInTheDocument()
  })
  it('takes faces that are someone else off them, with Undo, as one decision', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces/people/1?face=12' })
    const grid = await screen.findByRole('grid', { name: 'Faces of Ada' })
    await user.click(within(within(grid).getByRole('row', { name: 'Face 12' })).getByRole('checkbox'))
    await user.click(within(within(grid).getByRole('row', { name: 'Face 13' })).getByRole('checkbox'))
    expect(screen.queryByRole('button', { name: 'Show in its photo' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Not Ada (2)' }))
    expect(await screen.findByText(/Took 2\sfaces off Ada\./)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'In its photo' })).not.toBeInTheDocument() // the open face went with them
    expect(within(grid).queryByRole('row', { name: 'Face 12' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Undo' }))
    expect(await within(grid).findByRole('row', { name: 'Face 12' })).toBeInTheDocument()
    await user.click(within(within(grid).getByRole('row', { name: 'Face 12' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Not Ada (1)' }))
    await commits(unmount, 'assign 12 to 1 rejected by strata-models')
  })
  it('says when only the best 500 faces are shown, and shows them all on asking', async () => {
    faceMock.people[0].faces = 600
    const { user } = setup(<App />, { route: '/faces/people/1' })
    expect(await screen.findByText(/Showing the best 3 of 600\sfaces\./)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show them all' }))
    await waitFor(() => expect(faceMock.calls).toContain('faces of 1 up to 5000'))
    expect(await screen.findByText(/Showing the best 3 of 600\sfaces\./)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Show them all' })).not.toBeInTheDocument()
  })
  it('renames them, and refuses an empty name', async () => {
    const { user } = setup(<App />, { route: '/faces/people/2' })
    expect(await screen.findByRole('heading', { name: 'Person 2', level: 1 })).toBeInTheDocument()
    const name = await screen.findByRole('textbox', { name: 'Name' })
    await user.click(screen.getByRole('button', { name: 'Rename' }))
    expect(await screen.findByText('A name cannot be empty.')).toBeInTheDocument()
    await user.type(name, 'Bo')
    await user.click(screen.getByRole('button', { name: 'Rename' }))
    expect(await screen.findByRole('heading', { name: 'Bo', level: 1 })).toBeInTheDocument()
    expect(faceMock.calls).toContain('rename 2 Bo')
  })
  it("says the service's reason when a rename fails", async () => {
    faceMock.fail = 'rename'
    const { user } = setup(<App />, { route: '/faces/people/1' })
    await user.type(await screen.findByRole('textbox', { name: 'Name' }), 'x')
    await user.click(screen.getByRole('button', { name: 'Rename' }))
    expect(await screen.findAllByText(/the face store is busy/)).not.toHaveLength(0)
  })
  it('teaches where faces come from when none are confirmed, and says when there is no such person', async () => {
    faceMock.faces[2] = []
    const { unmount } = setup(<App />, { route: '/faces/people/2' })
    expect(await screen.findByText('No faces confirmed yet')).toBeInTheDocument()
    unmount()
    setup(<App />, { route: '/faces/people/99' })
    expect(await screen.findAllByRole('alert')).not.toHaveLength(0)
  })
})

describe('Faces: unnamed groups', () => {
  it('names a group as a new person, all its faces chosen to start with', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces?tab=groups' })
    expect(await screen.findByRole('heading', { name: /Group 1 of 2: 3\sfaces/ })).toBeInTheDocument()
    const grid = screen.getByRole('grid', { name: 'Faces in group 1' })
    expect(within(within(grid).getByRole('row', { name: 'Face 31' })).getByRole('checkbox')).toBeChecked()
    await user.click(within(within(grid).getByRole('row', { name: 'Face 33' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Name 2' }))
    expect(await screen.findByText('Give the person a name.')).toBeInTheDocument()
    await user.type(screen.getByRole('textbox', { name: 'A new person' }), 'Dee')
    await user.click(screen.getByRole('button', { name: 'Name 2' }))
    expect(await screen.findByText(/Named 2\sfaces Dee\./)).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: /Group 1 of 2: 1\sface$/ })).toBeInTheDocument() // the unticked face stays to be decided
    expect(within(screen.getByRole('grid', { name: 'Faces in group 1' })).getByRole('row', { name: 'Face 33' })).toBeInTheDocument()
    await commits(unmount, 'assign 31,32 to 53 confirmed by strata-models')
    expect(faceMock.calls).toContain('create Dee')
    expect(faceMock.calls).toContain('cluster by strata-models')
  })
  it('adds a group to someone named already, steps between groups, and marks one as not faces', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces?tab=groups' })
    await screen.findByRole('heading', { name: /Group 1 of 2: 3\sfaces/ })
    expect(screen.getByRole('button', { name: 'Previous group' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next group' }))
    expect(screen.getByRole('heading', { name: /Group 2 of 2: 2\sfaces/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Previous group' }))
    expect(screen.getByRole('heading', { name: /Group 1 of 2: 3\sfaces/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next group' }))
    expect(screen.getByRole('button', { name: 'Add to them' })).toBeDisabled()
    await user.selectOptions(screen.getByRole('combobox', { name: 'Or someone named already' }), 'Cy')
    await user.click(screen.getByRole('button', { name: 'Add to Cy' }))
    expect(await screen.findByText(/Added 2\sfaces to Cy\./)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Group 1 of 1: 3\sfaces/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Not faces' })) // scheduling a second decision commits the first
    await waitFor(() => expect(faceMock.calls).toContain('assign 41,42 to 3 confirmed by strata-models'))
    expect(await screen.findByText(/Marked 3\sfaces as not faces\./)).toBeInTheDocument()
    await commits(unmount, 'assign 31,32,33 to null rejected by strata-models')
  })
  it('says every face has been decided when no group is left', async () => {
    faceMock.groups = []
    setup(<App />, { route: '/faces?tab=groups' })
    expect(await screen.findByText('Every face has been decided')).toBeInTheDocument()
  })
  it('brings the group back and says so when a decision does not go through', async () => {
    faceMock.fail = 'assign'
    const { user } = setup(<App />, { route: '/faces?tab=groups' })
    await screen.findByRole('heading', { name: /Group 1 of 2: 3\sfaces/ })
    await user.click(screen.getByRole('button', { name: 'Not faces' }))
    await screen.findByRole('heading', { name: /Group 1 of 1: 2\sfaces/ })
    await user.click(screen.getByRole('button', { name: 'Not faces' })) // commits the first, which fails
    expect(await screen.findByRole('heading', { name: /Group 1 of 1: 3\sfaces/ })).toBeInTheDocument() // the failed one is back
    await user.click(screen.getByRole('button', { name: 'Undo' }))
    expect(await screen.findByText(/3\sfaces are back in the group/)).toBeInTheDocument()
  })
})

describe('Faces: possible duplicates', () => {
  it('merges two people into the one with more faces', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces?tab=duplicates' })
    const list = await screen.findByRole('list', { name: 'Possible duplicates' })
    expect(within(list).getByText('70% alike')).toBeInTheDocument()
    expect(within(list).getByText(/3\sfaces · bright\-horizons/)).toBeInTheDocument()
    await user.click(within(list).getByRole('button', { name: 'Merge Ada and Cy' }))
    expect(await screen.findByText('Merged Cy into Ada.')).toBeInTheDocument()
    expect(await screen.findByText('Nobody looks like anyone else')).toBeInTheDocument()
    await commits(unmount, 'merge 3 into 1 by strata-models')
  })
  it('keeps two people apart, and merges into the second when it has more faces', async () => {
    faceMock.suggestions = [{ a: 3, b: 1, distance: 0.5, apps_a: [], apps_b: [] }, { a: 2, b: 1, distance: 0.55, apps_a: [], apps_b: [] }]
    const { user, unmount } = setup(<App />, { route: '/faces?tab=duplicates' })
    const list = await screen.findByRole('list', { name: 'Possible duplicates' })
    await user.click(within(list).getAllByRole('button', { name: 'Different people' })[0])
    expect(await screen.findByText('Kept Cy and Ada apart.')).toBeInTheDocument()
    await user.click(within(list).getByRole('button', { name: 'Merge Person 2 and Ada' }))
    await waitFor(() => expect(faceMock.calls).toContain('not-same 3 1'))
    expect(await screen.findByText('Merged Person 2 into Ada.')).toBeInTheDocument()
    await commits(unmount, 'merge 2 into 1 by strata-models')
  })
})

describe('Faces: tags', () => {
  it('lists the tags and keeps one out of searches until asked for', async () => {
    const { user } = setup(<App />, { route: '/faces?tab=tags' })
    const table = await screen.findByRole('grid', { name: 'Tags' })
    expect(within(table).getByRole('switch', { name: /nursery/ })).toBeChecked()
    await user.click(within(table).getByRole('switch', { name: /bright-horizons/ }))
    await waitFor(() => expect(faceMock.calls).toContain('tag bright-horizons true'))
    await waitFor(() => expect(within(table).getByRole('switch', { name: /bright-horizons/ })).toBeChecked())
  })
  it("says the service's reason when a tag cannot be changed, and when there are none", async () => {
    faceMock.fail = 'tag'
    const { user, unmount } = setup(<App />, { route: '/faces?tab=tags' })
    await user.click(within(await screen.findByRole('grid', { name: 'Tags' })).getByRole('switch', { name: /nursery/ }))
    expect(await screen.findByText('the face store is busy')).toBeInTheDocument()
    unmount()
    faceMock.tags = []
    setup(<App />, { route: '/faces?tab=tags' })
    expect(await screen.findByText('No tags yet')).toBeInTheDocument()
  })
})
