import { useState } from 'react'
import AppShell from './components/AppShell'
import LandingPage from './components/LandingPage'
import ScannerPage from './components/ScannerPage'
import ContractAuditor from './components/ContractAuditor'
import Watchlist from './components/Watchlist'
import { useAlerts } from './hooks/useAlerts'
import { useTheme } from './hooks/useTheme'
import { useWatchlist } from './hooks/useWatchlist'

export default function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [page, setPage] = useState('scanner')
  const [selectedHash, setSelectedHash] = useState(null)
  const [auditAddress, setAuditAddress] = useState('')
  const [live, setLive] = useState(true)

  const { theme, toggle } = useTheme()
  const { alerts, status, updatedAt } = useAlerts(2000, !live)
  const watchlist = useWatchlist()

  const openAuditor = (address = '') => {
    setAuditAddress(address)
    setPage('auditor')
    setShowLanding(false)
  }

  if (showLanding) {
    return (
      <LandingPage
        alerts={alerts}
        onEnter={() => { setPage('scanner'); setShowLanding(false) }}
        onAudit={() => openAuditor()}
        theme={theme}
        onToggleTheme={toggle}
      />
    )
  }

  return (
    <AppShell
      current={page}
      onNavigate={setPage}
      onExit={() => setShowLanding(true)}
      status={status}
      updatedAt={updatedAt}
      counts={{ scanner: alerts.length, watchlist: watchlist.list.length }}
      live={live}
      onToggleLive={() => setLive((l) => !l)}
      theme={theme}
      onToggleTheme={toggle}
    >
      {page === 'scanner' && (
        <ScannerPage
          alerts={alerts}
          selectedHash={selectedHash}
          onSelect={setSelectedHash}
          onAudit={openAuditor}
          watched={watchlist.list}
          onWatch={watchlist.add}
        />
      )}
      {page === 'auditor' && <ContractAuditor initialAddress={auditAddress} />}
      {page === 'watchlist' && (
        <Watchlist
          watched={watchlist.list}
          alerts={alerts}
          onAdd={watchlist.add}
          onRemove={watchlist.remove}
        />
      )}
    </AppShell>
  )
}
