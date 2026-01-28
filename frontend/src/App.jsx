import { useState, useEffect } from 'react'
import LandingPage from './components/LandingPage'
import LiveTransactionScanner from './components/LiveTransactionScanner'
import FraudAlerts from './components/FraudAlerts'
import SmartContractAnalysis from './components/SmartContractAnalysis'
import WalletWatcher from './components/WalletWatcher'
import './App.css'

// --- ICONS ---
const ScannerIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
  </svg>
)

const AuditorIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
)

const WalletIcon = () => (
  <svg className="sidebar-nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="5" width="20" height="14" rx="2" />
    <line x1="2" y1="10" x2="22" y2="10" />
  </svg>
)

const BlockIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
  </svg>
)

const ReconLogo = () => (
  <svg className="sidebar-logo-icon" viewBox="0 0 40 40" fill="none">
    <circle cx="20" cy="20" r="18" stroke="url(#logoGradient)" strokeWidth="2" />
    <path d="M20 10v20M10 20h20" stroke="url(#logoGradient)" strokeWidth="2" strokeLinecap="square" />
    <rect x="14" y="14" width="12" height="12" stroke="url(#logoGradient)" strokeWidth="2" transform="rotate(45 20 20)" />
    <defs>
      <linearGradient id="logoGradient" x1="0" y1="0" x2="40" y2="40">
        <stop offset="0%" stopColor="#00f0ff" />
        <stop offset="100%" stopColor="#bd00ff" />
      </linearGradient>
    </defs>
  </svg>
)

function App() {
  const [alerts, setAlerts] = useState([])
  const [serverStatus, setServerStatus] = useState("Checking...")
  const [currentPage, setCurrentPage] = useState('transaction-scanner')
  const [latestBlock, setLatestBlock] = useState(null)

  // Start directly on dashboard for development speed (skip landing for now during dev, or keep it)
  // For verify we might want to see landing page but user asked for overhaul, so maybe landing page needs love too?
  // Let's keep existing flow: Landing first.
  const [showLanding, setShowLanding] = useState(true)

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/get-alerts`);
        const data = await res.json()

        setAlerts(data)
        setServerStatus("Online")

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

  const fraudCount = alerts.filter(tx => tx.prediction === 1 || tx.is_fraud).length
  const isOnline = serverStatus === "Online"

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} />
  }

  return (
    <div className="app-container">
      <div className="app-bg-grid" />

      {/* COMMAND BAR (Sidebar) */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <ReconLogo />
        </div>

        <nav className="sidebar-nav">
          <button
            className={`sidebar-nav-item ${currentPage === 'transaction-scanner' ? 'active' : ''}`}
            onClick={() => setCurrentPage('transaction-scanner')}
            data-tooltip="Live Scanner"
          >
            <ScannerIcon />
          </button>

          <button
            className={`sidebar-nav-item ${currentPage === 'contract-analysis' ? 'active' : ''}`}
            onClick={() => setCurrentPage('contract-analysis')}
            data-tooltip="Contract Auditor"
          >
            <AuditorIcon />
          </button>

          <button
            className={`sidebar-nav-item ${currentPage === 'wallet-watcher' ? 'active' : ''}`}
            onClick={() => setCurrentPage('wallet-watcher')}
            data-tooltip="Wallet Watcher"
          >
            <WalletIcon />
          </button>
        </nav>
      </aside>

      {/* MAIN CONTENT */}
      <div className="main-wrapper">
        <header className="stats-bar">
          <div className="stats-bar-left">
            <h2 style={{ fontSize: '18px', margin: 0, color: 'white' }}>
              {currentPage === 'transaction-scanner' && 'LIVE DETECTOR'}
              {currentPage === 'contract-analysis' && 'CONTRACT AUDIT'}
              {currentPage === 'wallet-watcher' && 'WALLET WATCHER'}
            </h2>
          </div>

          <div className="stats-bar-right">
            <div className="stat-item">
              <BlockIcon />
              <span className="stat-label">Block Height:</span>
              <span className="stat-value">{latestBlock ? latestBlock.toLocaleString() : 'Loading...'}</span>
            </div>

            <div className={`connection-status`}>
              <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
              <span className={`connection-text ${isOnline ? 'online' : 'offline'}`}>
                {isOnline ? 'Mainnet Connected' : 'Reconnecting...'}
              </span>
            </div>
          </div>
        </header>

        <main className="main-content">
          <div className="dashboard-grid">
            {currentPage === 'transaction-scanner' && (
              <>
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                  <LiveTransactionScanner transactions={alerts} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
                  <FraudAlerts allTransactions={alerts} />
                </div>
              </>
            )}

            {currentPage === 'contract-analysis' && (
              <div style={{ gridColumn: '1 / -1' }}>
                <SmartContractAnalysis />
              </div>
            )}

            {currentPage === 'wallet-watcher' && (
              <div style={{ gridColumn: '1 / -1' }}>
                <WalletWatcher />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App