import { ShieldCheck, House } from 'lucide-react'
import { relativeTime } from '../lib/format'

const STATUS = {
  online: { color: 'var(--risk-safe)', label: 'Live' },
  connecting: { color: 'var(--risk-med)', label: 'Connecting' },
  offline: { color: 'var(--risk-high)', label: 'Offline' },
}

// Desktop vertical nav. Hidden on mobile, where MobileNav takes over.
export default function Sidebar({ nav, current, onNavigate, onExit, status, count, updatedAt }) {
  const s = STATUS[status] || STATUS.connecting

  return (
    <aside className="hidden w-[216px] flex-none flex-col border-r border-line bg-surface md:flex">
      <div className="flex h-[54px] items-center gap-[9px] border-b border-line px-4">
        <span className="grid size-[26px] place-items-center rounded-[7px] border border-accent text-accent">
          <ShieldCheck size={15} />
        </span>
        <span className="text-[15px]" style={{ fontFamily: 'var(--font-heading)' }}>Recon</span>
        <button className="btn btn-ghost ml-auto px-1 py-0.5" onClick={onExit} aria-label="Back to site">
          <House size={14} />
        </button>
      </div>

      <nav className="flex flex-col gap-0.5 px-2.5 py-3">
        {nav.map(({ id, label, icon: Icon }) => {
          const active = current === id
          return (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              aria-current={active ? 'page' : undefined}
              className="rc-nav flex cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13.5px]"
              style={{
                background: active ? 'color-mix(in srgb, var(--color-accent) 16%, transparent)' : 'transparent',
                color: active ? 'var(--color-accent)' : 'color-mix(in srgb, var(--color-text) 70%, transparent)',
              }}
            >
              <Icon size={17} className="shrink-0" />
              {label}
              {id === 'auditor' ? null : (
                <span className="mono ml-auto text-[11px]" style={{ color: 'color-mix(in srgb, var(--color-text) 40%, transparent)' }}>
                  {count[id]}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      <div className="mt-auto p-3">
        <div className="rounded-md px-3 py-[11px]" style={{ background: 'var(--surface-2)' }}>
          <div className="flex items-center gap-[7px] text-[11px] uppercase tracking-[0.07em] text-muted">
            <span
              className={`size-[5px] rounded-full ${status === 'online' ? 'rc-pulse' : ''}`}
              style={{ background: s.color }}
              aria-hidden="true"
            />
            {s.label}
          </div>
          <div className="mono mt-[7px] text-[11px]" style={{ color: 'color-mix(in srgb, var(--color-text) 45%, transparent)' }}>
            {count.scanner.toLocaleString()} tx in window
          </div>
          <div className="mono mt-0.5 text-[11px]" style={{ color: 'color-mix(in srgb, var(--color-text) 45%, transparent)' }}>
            updated {updatedAt ? relativeTime(updatedAt) : '—'}
          </div>
        </div>
      </div>
    </aside>
  )
}

// Mobile bottom tab bar.
export function MobileNav({ nav, current, onNavigate }) {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-line bg-surface md:hidden">
      {nav.map(({ id, short, icon: Icon }) => {
        const active = current === id
        return (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            aria-current={active ? 'page' : undefined}
            className="flex flex-1 flex-col items-center gap-1 py-2.5 text-xs"
            style={{ color: active ? 'var(--color-accent)' : 'color-mix(in srgb, var(--color-text) 60%, transparent)' }}
          >
            <Icon size={20} />
            {short}
          </button>
        )
      })}
    </nav>
  )
}
