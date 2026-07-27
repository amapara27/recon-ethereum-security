import { ShieldCheck } from 'lucide-react'
import ThemeToggle from './ThemeToggle'

const MUTED = 'color-mix(in srgb, var(--color-text) 55%, transparent)'
const STATUS_COLOR = {
  online: 'var(--risk-safe)',
  connecting: 'var(--risk-med)',
  offline: 'var(--risk-high)',
}

export default function TopBar({ title, subtitle, status, live, onToggleLive, theme, onToggleTheme }) {
  const color = live ? STATUS_COLOR[status] || STATUS_COLOR.connecting : MUTED

  return (
    <header className="flex h-[54px] flex-none items-center gap-2.5 border-b border-line bg-app px-4 sm:px-5">
      {/* Brand shows here only on mobile — the sidebar is hidden below md */}
      <span className="grid size-[26px] place-items-center rounded-[7px] border border-accent text-accent md:hidden">
        <ShieldCheck size={15} />
      </span>
      <h1 className="m-0 text-[15px] tracking-[-0.01em]">{title}</h1>
      <span className="hidden text-xs text-muted sm:inline">{subtitle}</span>

      <div className="ml-auto flex items-center gap-2">
        <span className="hidden items-center gap-1.5 rounded-full border border-line px-2.5 py-[5px] text-xs lg:inline-flex">
          <span className="text-accent" aria-hidden="true">Ξ</span>
          Ethereum Mainnet
        </span>
        <button
          onClick={onToggleLive}
          aria-pressed={!live}
          className="rc-chip inline-flex cursor-pointer items-center gap-1.5 rounded-full border bg-transparent px-2.5 py-[5px] text-xs"
          style={{ borderColor: live ? `color-mix(in srgb, ${color} 45%, transparent)` : 'var(--color-divider)', color }}
        >
          <span
            className={`size-1.5 rounded-full ${live && status === 'online' ? 'rc-pulse' : ''}`}
            style={{ background: color }}
            aria-hidden="true"
          />
          {live ? 'Live' : 'Paused'}
        </button>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} size={30} />
      </div>
    </header>
  )
}
