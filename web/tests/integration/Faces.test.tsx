import { describe, expect, it } from 'vitest'
import App from '../../src/App'
import { faceMock } from '../utils/faces'
import { screen, setup, waitFor, within } from '../utils/render'

// A removal waits out its Undo window (8 s); leaving the screen commits it, which is how these tests see the request without waiting.
const commits = async (unmount: () => void, call: string) => { unmount(); await waitFor(() => expect(faceMock.calls).toContain(call)) }

describe('Faces: people', () => {
  it('lists everyone named, most faces first, and finds a person by name', async () => {
    const { user } = setup(<App />, { route: '/faces' })
    expect(await screen.findByText(/3\speople named, with 6\sfaces confirmed and 3 suggested to confirm\./)).toBeInTheDocument()
    const table = await screen.findByRole('grid', { name: 'People' })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map(r => within(r).getByRole('link').textContent)).toEqual(['Ada', 'Cy', 'Person 2'])
    expect(within(rows[0]).getByText('bright-horizons')).toBeInTheDocument()
    expect(within(rows[1]).getByRole('gridcell', { name: '3' })).toBeInTheDocument() // Cy's suggested faces
    expect(within(rows[2]).getAllByText('–')).toHaveLength(2) // nothing suggested, named in no app
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

describe('Faces: rounds of suggested faces', () => {
  it('offers the most certain first, every face chosen, each with its likeness, and confirms them', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByRole('heading', { name: /Is this Cy\? 3\sfaces suggested/ })).toBeInTheDocument()
    expect(await screen.findByText(/Nothing confirmed yet: these are from the tightest group of 3\ssuggestions/)).toBeInTheDocument()
    expect(screen.getByText(/2 are very close and 2 close\./)).toBeInTheDocument()
    const grid = await screen.findByRole('grid', { name: 'Is this Cy?' })
    expect(within(grid).getByText('90% alike')).toBeInTheDocument()
    expect(within(within(grid).getByRole('row', { name: /Face 61/ })).getByRole('checkbox')).toBeChecked()
    expect(faceMock.calls).toContain('review 3 up to 48')
    await user.click(screen.getByRole('button', { name: 'Confirm 3' }))
    expect(await screen.findByText(/Confirmed 3\sfaces as Cy\./)).toBeInTheDocument()
    expect(await screen.findByText('Nothing left to confirm')).toBeInTheDocument()
    await commits(unmount, 'assign 61,62,63 to 3 confirmed by strata-models')
  })
  it('says the unticked ones are not them, scored against confirmed faces once there are enough', async () => {
    faceMock.faces[3].push({ ...faceMock.faces[3][0], id: 17 })
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByText(/Scored against 3\sconfirmed faces\./)).toBeInTheDocument()
    const grid = await screen.findByRole('grid', { name: 'Is this Cy?' })
    await user.click(within(within(grid).getByRole('row', { name: /Face 63/ })).getByRole('checkbox'))
    expect(screen.getByRole('button', { name: 'Confirm 2' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Confirm 2, the other 1 are not Cy' }))
    expect(await screen.findByText(/Confirmed 2\sfaces as Cy, and 1\sface as not them\./)).toBeInTheDocument()
    unmount()
    await waitFor(() => expect(faceMock.calls).toContain('assign 61,62 to 3 confirmed by strata-models'))
    await waitFor(() => expect(faceMock.calls).toContain('assign 63 to 3 rejected by strata-models'))
  })
  it('opens a face of the round in its photo, and confirms none when all are unticked', async () => {
    const { user } = setup(<App />, { route: '/faces/people/3' })
    const grid = await screen.findByRole('grid', { name: 'Is this Cy?' })
    for (const n of [62, 63]) await user.click(within(within(grid).getByRole('row', { name: new RegExp(`Face ${n}`) })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Show in its photo' }))
    expect(await screen.findByRole('heading', { name: 'In its photo' })).toBeInTheDocument()
    await user.click(within(within(grid).getByRole('row', { name: /Face 61/ })).getByRole('checkbox'))
    expect(screen.getByRole('button', { name: 'Confirm 0' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Confirm 0, the other 3 are not Cy' }))
    expect(await screen.findByText(/Confirmed 0\sfaces as Cy, and 3\sfaces as not them\./)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Undo' })) // nothing left to send when the next test starts
  })
  it('takes a round back with Undo before it is sent', async () => {
    const { user } = setup(<App />, { route: '/faces/people/3' })
    await screen.findByRole('grid', { name: 'Is this Cy?' })
    await user.click(screen.getByRole('button', { name: 'Confirm 3' }))
    expect(await screen.findByText('Nothing left to confirm')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Undo' }))
    expect(await screen.findByRole('grid', { name: 'Is this Cy?' })).toBeInTheDocument()
    expect(faceMock.calls.filter(c => c.startsWith('assign'))).toEqual([])
  })
})

describe('Faces: every suggestion above a likeness, once a person has many confirmed faces', () => {
  it('keeps the rounds up to 40 confirmed faces and shows the picker from 41', async () => {
    faceMock.people[2].faces = 40
    const { unmount } = setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByRole('button', { name: 'Confirm 3' })).toBeInTheDocument()
    expect(screen.queryByRole('textbox', { name: 'Show every suggestion at least this alike' })).not.toBeInTheDocument()
    unmount()
    faceMock.people[2].faces = 41
    setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByRole('textbox', { name: 'Show every suggestion at least this alike' })).toBeInTheDocument()
  })
  it('shows every suggestion at the chosen likeness with counts at each, and confirms them all at once', async () => {
    faceMock.people[2].faces = 150
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    const grid = await screen.findByRole('grid', { name: 'Is this Cy?' })
    expect(faceMock.calls).toContain('review 3 up to 10000 at 0.7')
    expect(within(grid).getAllByRole('row')).toHaveLength(2) // 90% and 75% alike; the 60% one is below 70%
    const counts = screen.getByRole('list', { name: 'Suggestions at each likeness' })
    expect(within(counts).getByRole('button', { name: '90%+: 1' })).toBeInTheDocument()
    await user.click(within(counts).getByRole('button', { name: '50%+: 3' }))
    await waitFor(() => expect(faceMock.calls).toContain('review 3 up to 10000 at 0.5'))
    expect(await screen.findByRole('button', { name: 'Confirm all 3 at 50%+' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Confirm all 3 at 50%+' }))
    expect(await screen.findByText(/Confirmed 3\sfaces as Cy\./)).toBeInTheDocument()
    await commits(unmount, 'assign 61,62,63 to 3 confirmed by strata-models')
  })
  it('takes the likeness from the address, a typed one too, and says when none are that alike', async () => {
    faceMock.people[2].faces = 150
    const { user } = setup(<App />, { route: '/faces/people/3?min=95' })
    expect(await screen.findByText('None at 95% or more')).toBeInTheDocument()
    expect(screen.getByText('Lower the likeness to see more.')).toBeInTheDocument()
    const field = screen.getByRole('textbox', { name: 'Show every suggestion at least this alike' })
    await user.clear(field)
    await user.type(field, '80')
    await user.tab()
    await waitFor(() => expect(faceMock.calls).toContain('review 3 up to 10000 at 0.8'))
    await user.clear(field)
    await user.tab() // an empty field keeps the last likeness
  })
})

describe('Faces: suggested faces of a person', () => {
  it('shows the suggested faces and confirms the chosen ones, with Undo', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByText(/2\sfaces confirmed, 3 suggested, named in strata\-360\./)).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: 'Browse every suggestion (3)' }))
    expect(await screen.findByRole('heading', { name: /Every suggestion: 3\sfaces/ })).toBeInTheDocument()
    const grid = await screen.findByRole('grid', { name: 'Faces suggested for Cy' })
    expect(faceMock.calls).toContain('suggested of 3 from 0 up to 200')
    await user.click(within(within(grid).getByRole('row', { name: 'Face 61' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Show in its photo' }))
    expect(await screen.findByRole('heading', { name: 'In its photo' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Confirm selected (1)' }))
    expect(await screen.findByText(/Confirmed 1\sface as Cy\./)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'In its photo' })).not.toBeInTheDocument() // the open face went with it
    expect(within(grid).queryByRole('row', { name: 'Face 61' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Undo' }))
    expect(await within(grid).findByRole('row', { name: 'Face 61' })).toBeInTheDocument()
    await user.click(within(within(grid).getByRole('row', { name: 'Face 62' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Confirm selected (1)' }))
    await commits(unmount, 'assign 62 to 3 confirmed by strata-models')
  })
  it('says a chosen face is not this person, recorded as a rejection', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    await user.click(await screen.findByRole('button', { name: 'Browse every suggestion (3)' }))
    const grid = await screen.findByRole('grid', { name: 'Faces suggested for Cy' })
    await user.click(within(within(grid).getByRole('row', { name: 'Face 63' })).getByRole('checkbox'))
    await user.click(screen.getByRole('button', { name: 'Not this person (1)' }))
    expect(await screen.findByText(/Took 1\sface off Cy's suggestions\./)).toBeInTheDocument()
    await commits(unmount, 'assign 63 to 3 rejected by strata-models')
  })
  it('confirms every suggested face at once, gathered from every page', async () => {
    const { user, unmount } = setup(<App />, { route: '/faces/people/3' })
    await user.click(await screen.findByRole('button', { name: 'Browse every suggestion (3)' }))
    await screen.findByRole('grid', { name: 'Faces suggested for Cy' })
    await user.click(screen.getByRole('button', { name: 'Confirm all 3' }))
    expect(await screen.findByText(/Confirmed 3\sfaces as Cy\./)).toBeInTheDocument()
    expect(await screen.findByText('Nothing left to confirm on this page')).toBeInTheDocument()
    expect(faceMock.calls).toContain('suggested of 3 from 0 up to 5000')
    await commits(unmount, 'assign 61,62,63 to 3 confirmed by strata-models')
  })
  it('pages through a long list of suggestions', async () => {
    faceMock.people[2].suggested = 450
    const { user } = setup(<App />, { route: '/faces/people/3' })
    await user.click(await screen.findByRole('button', { name: 'Browse every suggestion (450)' }))
    expect(await screen.findByText('Page 1 of 3')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(await screen.findByText('Page 2 of 3')).toBeInTheDocument()
    await waitFor(() => expect(faceMock.calls).toContain('suggested of 3 from 200 up to 200'))
    expect(await screen.findByText('Nothing left to confirm on this page')).toBeInTheDocument()
    expect(screen.getByText('Go back a page, or come back once the list has refreshed.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next page' }))
    expect(await screen.findByText('Page 3 of 3')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Previous page' }))
    expect(await screen.findByText('Page 2 of 3')).toBeInTheDocument()
  })
  it('points at the suggestions when nothing is confirmed yet, and hides the section when there are none', async () => {
    faceMock.faces[3] = []
    const { unmount } = setup(<App />, { route: '/faces/people/3' })
    expect(await screen.findByText('Confirm the suggested faces below.')).toBeInTheDocument()
    unmount()
    setup(<App />, { route: '/faces/people/1' })
    await screen.findByRole('grid', { name: 'Faces of Ada' })
    expect(screen.queryByRole('heading', { name: /suggested/ })).not.toBeInTheDocument()
  })
  it("says the service's reason when the suggestions cannot be gathered", async () => {
    const { user } = setup(<App />, { route: '/faces/people/3' })
    await user.click(await screen.findByRole('button', { name: 'Browse every suggestion (3)' }))
    await screen.findByRole('grid', { name: 'Faces suggested for Cy' })
    faceMock.suggested[3] = Array.from({ length: 5000 }, (_, i) => ({ ...faceMock.suggested[3][0], id: 1000 + i }))
    faceMock.fail = 'faces'
    await user.click(screen.getByRole('button', { name: 'Confirm all 3' }))
    expect(await screen.findAllByText(/the face store is busy/)).not.toHaveLength(0)
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
    expect(faceMock.calls).toContain('cluster by strata-models without suggested')
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
