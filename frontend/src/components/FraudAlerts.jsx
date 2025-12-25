import { useState, useEffect, useRef } from 'react'

// Utility: format relative time
function getRelativeTime(timestamp) {
  const now = Date.now()
  const diff = now - new Date(timestamp).getTime()
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)

  if (seconds < 60) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

// Risk Ring Component using SVG
function RiskRing({ score }) {
  const percentage = score * 100
  const radius = 20
  const strokeWidth = 4
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference

  // Color based on score
  const getColor = () => {
    if (percentage < 40) return { stroke: '#00ff88', glow: 'rgba(0, 255, 136, 0.4)' }
    if (percentage < 70) return { stroke: '#ffa502', glow: 'rgba(255, 165, 2, 0.4)' }
    return { stroke: '#ff4757', glow: 'rgba(255, 71, 87, 0.4)' }
  }

  const colors = getColor()

  return (
    <div className="risk-ring-container">
      <svg className="risk-ring" width="48" height="48" viewBox="0 0 48 48">
        {/* Background ring */}
        <circle
          cx="24"
          cy="24"
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
        />
        {/* Progress ring */}
        <circle
          cx="24"
          cy="24"
          r={radius}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            filter: `drop-shadow(0 0 4px ${colors.glow})`,
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%'
          }}
        />
      </svg>
      <span className="risk-ring-value" style={{ color: colors.stroke }}>
        {percentage.toFixed(0)}
      </span>
    </div>
  )
}

// Empty State Icon
function ShieldIcon() {
  return (
    <svg className="empty-state-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  )
}

// External Link Icon
function ExternalLinkIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  )
}

function FraudAlerts({ allTransactions }) {
  const [displayCount, setDisplayCount] = useState(20)
  const [newAlertIds, setNewAlertIds] = useState(new Set())
  const scrollContainerRef = useRef(null)
  const prevAlertsRef = useRef([])

  // Filter only fraudulent transactions (probability >= 0.3)
  const fraudulentTx = allTransactions.filter(tx => {
    const txTime = new Date(tx.timestamp).getTime()
    const hoursAgo24 = Date.now() - (24 * 60 * 60 * 1000)
    return tx.probability >= 0.3 && txTime >= hoursAgo24
  })

  // Track new alerts for animation
  useEffect(() => {
    const prevIds = new Set(prevAlertsRef.current.map(tx => tx.tx_hash))
    const newIds = fraudulentTx
      .filter(tx => !prevIds.has(tx.tx_hash))
      .map(tx => tx.tx_hash)

    if (newIds.length > 0) {
      setNewAlertIds(new Set(newIds))
      // Clear new status after animation
      const timer = setTimeout(() => setNewAlertIds(new Set()), 3000)
      return () => clearTimeout(timer)
    }

    prevAlertsRef.current = fraudulentTx
  }, [fraudulentTx])

  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container

      // Load more when scrolled to 80% of the container
      if (scrollTop + clientHeight >= scrollHeight * 0.8) {
        setDisplayCount(prev => Math.min(prev + 20, fraudulentTx.length))
      }
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [fraudulentTx.length])

  // Get severity class based on probability
  const getSeverityClass = (probability) => {
    if (probability >= 0.7) return 'severity-high'
    if (probability >= 0.5) return 'severity-medium'
    return 'severity-low'
  }

  if (fraudulentTx.length === 0) {
    return (
      <div className="card fraud-panel">
        <h3>Fraud Alerts</h3>
        <div className="alerts-empty-state">
          <ShieldIcon />
          <div className="alerts-empty-title">All Clear</div>
          <div className="alerts-empty-text">No fraudulent transactions detected in the last 24 hours</div>
        </div>
      </div>
    )
  }

  return (
    <div className="card fraud-panel">
      <h3>Fraud Alerts</h3>
      <div className="fraud-stats">
        <span className="fraud-count">{fraudulentTx.length}</span>
        <span className="fraud-label">alert{fraudulentTx.length !== 1 ? 's' : ''} detected (24h)</span>
      </div>
      <div className="alerts-scroll" ref={scrollContainerRef}>
        <div className="alerts-list">
          {fraudulentTx.slice(0, displayCount).map((alert, index) => (
            <div
              key={alert.tx_hash}
              className={`alert-card ${getSeverityClass(alert.probability)} ${newAlertIds.has(alert.tx_hash) ? 'alert-new' : ''} ${index % 2 === 1 ? 'alert-alt' : ''}`}
            >
              <div className="alert-card-left">
                <RiskRing score={alert.probability} />
              </div>

              <div className="alert-card-content">
                <div className="alert-card-header">
                  <span className="alert-address font-mono">
                    {alert.address.substring(0, 6)}...{alert.address.substring(38)}
                  </span>
                  <span className="alert-time">{getRelativeTime(alert.timestamp)}</span>
                </div>

                <div className="alert-card-details">
                  {alert.value && (
                    <div className="alert-value">
                      <span className="alert-value-amount">{parseFloat(alert.value).toFixed(4)}</span>
                      <span className="alert-value-label">ETH</span>
                    </div>
                  )}
                  <span className={`alert-severity-badge ${getSeverityClass(alert.probability)}`}>
                    {alert.probability >= 0.7 ? 'High Risk' : alert.probability >= 0.5 ? 'Medium Risk' : 'Elevated'}
                  </span>
                </div>
              </div>

              <a
                href={`https://etherscan.io/tx/0x${alert.tx_hash}`}
                target="_blank"
                rel="noreferrer"
                className="alert-card-action"
              >
                <span>View</span>
                <ExternalLinkIcon />
              </a>
            </div>
          ))}
        </div>

        {displayCount < fraudulentTx.length && (
          <div className="load-more-indicator">
            Scroll for more ({displayCount} of {fraudulentTx.length})
          </div>
        )}
      </div>
    </div>
  )
}

export default FraudAlerts
