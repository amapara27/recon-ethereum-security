import { Radar, FileSearch, Eye } from 'lucide-react'
import Sidebar, { MobileNav } from './Sidebar'
import TopBar from './TopBar'

const NAV = [
  { id: 'scanner', label: 'Live Scanner', short: 'Scanner', icon: Radar },
  { id: 'auditor', label: 'Contract Auditor', short: 'Auditor', icon: FileSearch },
  { id: 'watchlist', label: 'Watchlist', short: 'Watchlist', icon: Eye },
]

export default function AppShell({ current, onNavigate, status, theme, onToggleTheme, children }) {
  const title = NAV.find((n) => n.id === current)?.label ?? 'Recon'
  return (
    <div className="flex min-h-dvh bg-app text-ink">
      <Sidebar nav={NAV} current={current} onNavigate={onNavigate} />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar title={title} status={status} theme={theme} onToggleTheme={onToggleTheme} />
        <main className="flex-1 p-4 pb-24 md:pb-6 lg:p-6">{children}</main>
      </div>
      <MobileNav nav={NAV} current={current} onNavigate={onNavigate} />
    </div>
  )
}
