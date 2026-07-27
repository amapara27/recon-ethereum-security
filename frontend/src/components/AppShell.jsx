import { Radar, FileSearch, Eye } from 'lucide-react'
import Sidebar, { MobileNav } from './Sidebar'
import TopBar from './TopBar'

const NAV = [
  { id: 'scanner', label: 'Live scanner', short: 'Scanner', icon: Radar, sub: '· scoring every transfer at head' },
  { id: 'auditor', label: 'Contract auditor', short: 'Auditor', icon: FileSearch, sub: '· verified Solidity review' },
  { id: 'watchlist', label: 'Watchlist', short: 'Watchlist', icon: Eye, sub: '· score drift on pinned addresses' },
]

export default function AppShell({ current, onNavigate, onExit, status, updatedAt, counts, live, onToggleLive, theme, onToggleTheme, children }) {
  const page = NAV.find((n) => n.id === current) ?? NAV[0]

  return (
    <div className="flex h-dvh overflow-hidden bg-app text-ink">
      <Sidebar
        nav={NAV}
        current={current}
        onNavigate={onNavigate}
        onExit={onExit}
        status={status}
        count={counts}
        updatedAt={updatedAt}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          title={page.label}
          subtitle={page.sub}
          status={status}
          live={live}
          onToggleLive={onToggleLive}
          theme={theme}
          onToggleTheme={onToggleTheme}
        />
        <main className="rc-scroll flex-1 overflow-auto px-4 pb-24 pt-4 sm:px-5 md:pb-7">{children}</main>
      </div>
      <MobileNav nav={NAV} current={current} onNavigate={onNavigate} />
    </div>
  )
}
