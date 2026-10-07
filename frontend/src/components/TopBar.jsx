import { Pause, Play } from 'lucide-react'
import ThemeToggle from './ThemeToggle'
import Wordmark from './Wordmark'
import { relativeTime } from '../lib/format'

// The shell band is lab green in both themes, so its status colours are fixed for that ground.
const STATUS = {
  online: { color: '#8fe3c8', label: 'Live' },
  connecting: { color: '#f2c46b', label: 'Connecting' },
  offline: { color: '#ff9e90', label: 'Offline' },
}

export default function TopBar({ nav, current, counts, onNavigate, onExit, status, updatedAt, live, onToggleLive, theme, onToggleTheme }) {
  const s = STATUS[status] || STATUS.connecting

  const tabs = nav.map(({ id, label }) => {
    const active = current === id
    return (
      <button
        key={id}
        onClick={() => onNavigate(id)}
        aria-current={active ? 'page' : undefined}
        className={`flex h-full flex-1 cursor-pointer items-center justify-center gap-2 border-b-2 bg-transparent px-3 text-[14px] font-semibold md:flex-none ${
          active ? 'border-on-shell text-on-shell' : 'border-transparent text-on-shell-2 hover:text-on-shell'
        }`}
      >
        {label}
        {counts[id] != null && <span className="mono text-[12px] font-normal opacity-80">{counts[id].toLocaleString()}</span>}
      </button>
    )
  })

  return (
    <header className="flex-none bg-shell text-on-shell">
      <div className="mx-auto flex h-[52px] max-w-[1440px] items-center gap-2 px-4 sm:px-6">
        <button onClick={onExit} className="mr-4 cursor-pointer bg-transparent text-on-shell" aria-label="Recon, back to the overview">
          <Wordmark />
        </button>
        <nav className="hidden h-full items-stretch md:flex" aria-label="Sections">{tabs}</nav>

        <div className="ml-auto flex items-center gap-1.5">
          <span className="mr-2 hidden text-[12.5px] text-on-shell-2 lg:inline">Ethereum mainnet</span>
          <button
            onClick={onToggleLive}
            aria-pressed={!live}
            title={live ? 'Pause the feed' : 'Resume the feed'}
            className="flex h-8 cursor-pointer items-center gap-2 rounded-[3px] border border-on-shell-2/40 bg-transparent px-2.5 text-[12.5px] font-semibold text-on-shell hover:border-on-shell"
          >
            <span
              className={`size-[7px] rounded-full ${live && status === 'online' ? 'pulse' : ''}`}
              style={{ background: live ? s.color : 'var(--on-shell-2)' }}
              aria-hidden="true"
            />
            {live ? s.label : 'Paused'}
            <span className="hidden font-normal text-on-shell-2 sm:inline">
              {updatedAt ? relativeTime(updatedAt) : '—'}
            </span>
            {live ? <Pause size={13} aria-hidden="true" /> : <Play size={13} aria-hidden="true" />}
          </button>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} onShell />
        </div>
      </div>
      <nav className="flex h-11 border-t border-on-shell-2/20 md:hidden" aria-label="Sections">{tabs}</nav>
    </header>
  )
}
