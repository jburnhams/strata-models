import { describe, expect, it } from 'vitest'
import { parse, href } from '../../src/route'
import { helloUrl, withBase } from '../../src/api'
import { describeCount } from '../../src/components/GreetingPage'

describe('routes', () => {
  it('parses greeting and settings pages', () => {
    expect(parse('')).toEqual({ page: 'greeting' })
    expect(parse('#/')).toEqual({ page: 'greeting' })
    expect(parse('#/settings')).toEqual({ page: 'settings' })
    expect(href.greeting()).toBe('#/')
    expect(href.settings()).toBe('#/settings')
  })
})

describe('api helpers', () => {
  it('builds URLs with query parameters and respects withBase', () => {
    expect(withBase('/api/health')).toBe('/api/health')
    expect(withBase('relative')).toBe('relative')
    expect(helloUrl('Alice')).toBe('/api/hello?name=Alice')
    expect(helloUrl('Alice & Bob', 'Hi')).toBe('/api/hello?name=Alice+%26+Bob&greeting=Hi')
  })
})

describe('describeCount', () => {
  it('formats zero, singular and plural counts', () => {
    expect(describeCount(0)).toBe('No greetings recorded yet.')
    expect(describeCount(1)).toBe('1 greeting recorded')
    expect(describeCount(3)).toBe('3 greetings recorded')
  })
})
