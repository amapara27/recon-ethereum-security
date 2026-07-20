import { useState } from 'react'
import AppShell from './components/AppShell'
import LandingPage from './components/LandingPage'
import ScannerPage from './components/ScannerPage'
import ContractAuditor from './components/ContractAuditor'
import Watchlist from './components/Watchlist'
import { useAlerts } from './hooks/useAlerts'
import { useTheme } from './hooks/useTheme'

export default function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [page, setPage] = useState('scanner')
  const { theme, toggle } = useTheme()
  const { alerts, status } = useAlerts()

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} theme={theme} onToggleTheme={toggle} />
  }

  return (
    <AppShell current={page} onNavigate={setPage} status={status} theme={theme} onToggleTheme={toggle}>
      {page === 'scanner' && <ScannerPage alerts={alerts} />}
      {page === 'auditor' && <ContractAuditor />}
      {page === 'watchlist' && <Watchlist />}
    </AppShell>
  )
}
