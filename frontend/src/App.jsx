import { useState, useEffect } from 'react'
import LandingPage from './components/LandingPage'
import LiveTransactionScanner from './components/LiveTransactionScanner'
import FraudAlerts from './components/FraudAlerts'
import SmartContractAnalysis from './components/SmartContractAnalysis'
import WalletWatcher from './components/WalletWatcher'
import './App.css'

// Icon components for sidebar
const ScannerIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7V5a2 2 0 0 1 2-2h2" />
    <path d="M17 3h2a2 2 0 0 1 2 2v2" />
    <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
    <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
    <line x1="7" y1="12" x2="17" y2="12" />
  </svg>
)

const AuditorIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <polyline points="10 9 9 9 8 9" />
  </svg>
)

const WalletIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4" />
    <path d="M3 5v14a2 2 0 0 0 2 2h16v-5" />
    <path d="M18 12a2 2 0 0 0 0 4h4v-4z" />
  </svg>
)

const AlertIcon = () => (
  <svg className="alert-badge-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
)

const BlockIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
  </svg>
)

const ReconLogo = () => (
  <svg className="sidebar-logo-icon" viewBox="0 0 40 40" fill="none">
    <circle cx="20" cy="20" r="18" stroke="url(#logoGradient)" strokeWidth="2" fill="none" />
    <path d="M12 20h16M20 12v16M14 14l12 12M26 14L14 26" stroke="url(#logoGradient)" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="20" cy="20" r="4" fill="url(#logoGradient)" />
    <defs>
      <linearGradient id="logoGradient" x1="0" y1="0" x2="40" y2="40">
        <stop offset="0%" stopColor="#00d4ff" />
        <stop offset="100%" stopColor="#00ff88" />
      </linearGradient>
    </defs>
  </svg>
)

function App() {
  const [alerts, setAlerts] = useState([])
  const [serverStatus, setServerStatus] = useState("Checking...")
  const [currentPage, setCurrentPage] = useState('transaction-scanner')
  const [showLanding, setShowLanding] = useState(true)
  const [latestBlock, setLatestBlock] = useState(null)

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const apiUrl = "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/get-alerts`);
        const data = await res.json()

        setAlerts(data)
        setServerStatus("Online")

        // Extract latest block from transactions if available
        if (data && data.length > 0 && data[0].block_number) {
          setLatestBlock(data[0].block_number)
        }
      } catch (err) {
        console.log("API Error:", err)
        setServerStatus("Offline")
      }
    }

    fetchAlerts()
    const intervalId = setInterval(fetchAlerts, 2000)
    return () => clearInterval(intervalId)
  }, [])

  // Count fraud alerts
  const fraudCount = alerts.filter(tx => tx.prediction === 1 || tx.is_fraud).length

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} />
  }

  const isOnline = serverStatus === "Online"

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <ReconLogo />
          <span className="sidebar-logo-text">RECON</span>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`sidebar-nav-item ${currentPage === 'transaction-scanner' ? 'active' : ''}`}
            onClick={() => setCurrentPage('transaction-scanner')}
            data-tooltip="Scanner"
          >
            <ScannerIcon />
            <span className="sidebar-nav-label">Scanner</span>
          </button>

          <button
            className={`sidebar-nav-item ${currentPage === 'contract-analysis' ? 'active' : ''}`}
            onClick={() => setCurrentPage('contract-analysis')}
            data-tooltip="Auditor"
          >
            <AuditorIcon />
            <span className="sidebar-nav-label">Contract Auditor</span>
          </button>

          <button
            className={`sidebar-nav-item ${currentPage === 'wallet-watcher' ? 'active' : ''}`}
            onClick={() => setCurrentPage('wallet-watcher')}
            data-tooltip="Wallet"
          >
            <WalletIcon />
            <span className="sidebar-nav-label">Wallet Watcher</span>
          </button>
        </nav>
      </aside>

      {/* Main Wrapper */}
      <div className="main-wrapper">
        {/* Top Stats Bar */}
        <header className="stats-bar">
          <div className="stats-bar-left">
            <div className="stat-item">
              <BlockIcon />
              <span className="stat-label">Block</span>
              <span className="stat-value">{latestBlock ? latestBlock.toLocaleString() : '---'}</span>
            </div>

            <div className="stat-divider" />

            <div className="connection-status">
              <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
              <span className={`connection-text ${isOnline ? 'online' : 'offline'}`}>
                {isOnline ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          <div className="stats-bar-right">
            <div className={`alert-badge ${fraudCount === 0 ? 'no-alerts' : ''}`}>
              <AlertIcon />
              <span>{fraudCount} {fraudCount === 1 ? 'Alert' : 'Alerts'}</span>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          <div className="dashboard-grid">
            {currentPage === 'transaction-scanner' && (
              <>
                <LiveTransactionScanner transactions={alerts} />
                <FraudAlerts allTransactions={alerts} />
              </>
            )}
            {currentPage === 'contract-analysis' && <SmartContractAnalysis />}
            {currentPage === 'wallet-watcher' && <WalletWatcher />}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App