import type { ReactNode } from 'react'

export function Crumbs({ items }: { items: { label: string; to?: string }[] }) {
  return (
    <nav aria-label="Breadcrumb" className="text-sm text-stone-500 mb-3 flex flex-wrap gap-1">
      {items.map((c, i) => (
        <span key={i} className="flex gap-1">
          {i > 0 && <span aria-hidden>/</span>}
          {c.to ? <a className="hover:underline" href={c.to}>{c.label}</a> : <span className="text-stone-800 dark:text-stone-200">{c.label}</span>}
        </span>
      ))}
    </nav>
  )
}

export function Err({ msg }: { msg: string | null }) { return msg ? <p role="alert" className="text-red-600 text-sm my-2">{msg}</p> : null }

export function Card({ children, to, className = '' }: { children: ReactNode; to?: string; className?: string }) {
  const cls = `block rounded-lg border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 p-3 ${to ? 'hover:border-blue-500' : ''} ${className}`
  return to ? <a href={to} className={cls}>{children}</a> : <div className={cls}>{children}</div>
}

export const btn = 'rounded-md bg-blue-600 text-white px-3 py-1.5 text-sm hover:bg-blue-700 disabled:opacity-50'
export const btnQuiet = 'rounded-md border border-stone-300 dark:border-stone-700 px-2 py-0.5 text-xs hover:border-blue-500'
export const input = 'rounded-md border border-stone-300 dark:border-stone-700 bg-transparent px-2 py-1.5 text-sm'
