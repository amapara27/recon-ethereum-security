import { ShieldCheck } from 'lucide-react'
import StatusPill from './StatusPill'
import ThemeToggle from './ThemeToggle'

export default function TopBar({ title, status, theme, onToggleTheme }) {
  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-line bg-app/85 px-4 backdrop-blur lg:px-6">
      {/* Brand shows here only on mobile (sidebar is hidden) */}
      <span className="grid size-8 place-items-center rounded-lg bg-accent text-accent-fg md:hidden">
        <ShieldCheck size={18} />
      </span>
      <h1 className="text-base font-semibold tracking-tight">{title}</h1>

      <div className="ml-auto flex items-center gap-2.5">
        <span className="hidden items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-xs text-muted sm:inline-flex">
          <span className="size-1.5 rounded-full bg-accent" aria-hidden="true" />
          Ethereum Mainnet
        </span>
        <StatusPill status={status} />
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  )
}
