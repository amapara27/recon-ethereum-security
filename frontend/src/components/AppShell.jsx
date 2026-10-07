import TopBar from './TopBar'

const NAV = [
  { id: 'scanner', label: 'Scanner', title: 'Live scanner' },
  { id: 'auditor', label: 'Auditor', title: 'Contract auditor' },
  { id: 'watchlist', label: 'Watchlist', title: 'Watchlist' },
]

export default function AppShell({ current, onNavigate, onExit, status, updatedAt, counts, live, onToggleLive, theme, onToggleTheme, children }) {
  const page = NAV.find((n) => n.id === current) ?? NAV[0]

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-app text-ink">
      <TopBar
        nav={NAV}
        current={current}
        counts={counts}
        onNavigate={onNavigate}
        onExit={onExit}
        status={status}
        updatedAt={updatedAt}
        live={live}
        onToggleLive={onToggleLive}
        theme={theme}
        onToggleTheme={onToggleTheme}
      />
      <main className="scroll flex-1 overflow-auto">
        <div className="mx-auto max-w-[1440px] px-4 pb-10 pt-5 sm:px-6">
          <h1 className="mb-4 text-[22px]">{page.title}</h1>
          {children}
        </div>
      </main>
    </div>
  )
}
