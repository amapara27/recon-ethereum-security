import { useState, useEffect } from 'react'
import DisplayAlerts from './components/DisplayAlerts'
import './App.css'

function App() {
  const [alerts, setAlerts] = useState([])
  const [serverStatus, setServerStatus] = useState("Checking...")

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/get-alerts") 
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
        <span className={`status-badge ${serverStatus.includes("Online") ? "status-ok" : "status-err"}`}>
          {serverStatus}
        </span>
      </header>

      <main>
        <DisplayAlerts alerts={alerts} />
      </main>
    </div>
  )
}

export default App