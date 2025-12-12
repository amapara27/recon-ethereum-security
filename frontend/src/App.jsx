import { useState, useEffect } from 'react'
import LiveTransactionScanner from './components/LiveTransactionScanner'
import FraudAlerts from './components/FraudAlerts'
import SmartContractAnalysis from './components/SmartContractAnalysis'
import WalletWatcher from './components/WalletWatcher'
import './App.css'

function App() {
  const [alerts, setAlerts] = useState([])
  const [serverStatus, setServerStatus] = useState("Checking...")
  const [currentPage, setCurrentPage] = useState('transaction-scanner')

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        // const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000/api/get-alerts";
        const apiUrl = "http://localhost:8000";
        const res = await fetch(`${apiUrl}/api/get-alerts`);
        const data = await res.json()

        setAlerts(data)
        setServerStatus("Online 🟢")
      } catch (err) {
        console.log("API Error:", err)
        setServerStatus("Offline 🔴")
      }
    }

    fetchAlerts()
    const intervalId = setInterval(fetchAlerts, 2000)
    return () => clearInterval(intervalId)
  }, [])


  return (
    <div className="app-container">
      <header className="dashboard-header">
        <h1>🛡️ Recon Security Dashboard</h1>
        <nav className="nav-buttons">
          <button
            className={`nav-btn ${currentPage === 'transaction-scanner' ? 'active' : ''}`}
            onClick={() => setCurrentPage('transaction-scanner')}
          >
            Transaction Scanner
          </button>
          <button
            className={`nav-btn ${currentPage === 'contract-analysis' ? 'active' : ''}`}
            onClick={() => setCurrentPage('contract-analysis')}
          >
            Smart Contract Analysis
          </button>
          <button
            className={`nav-btn ${currentPage === 'wallet-watcher' ? 'active' : ''}`}
            onClick={() => setCurrentPage('wallet-watcher')}
          >
            Wallet Watcher
          </button>
        </nav>
        <span className={`status-badge ${serverStatus.includes("Online") ? "status-ok" : "status-err"}`}>
          {serverStatus}
        </span>
      </header>

      <main className="dashboard-grid">
        {currentPage === 'transaction-scanner' && (
          <>
            <LiveTransactionScanner transactions={alerts} />
            <FraudAlerts allTransactions={alerts} />
          </>
        )}
        {currentPage === 'contract-analysis' && <SmartContractAnalysis />}
        {currentPage === 'wallet-watcher' && <WalletWatcher />}
      </main>
    </div>
  )
}

export default App