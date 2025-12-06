import { useState, useEffect, useRef } from 'react'

function FraudAlerts({ allTransactions }) {
  const [displayCount, setDisplayCount] = useState(20)
  const scrollContainerRef = useRef(null)

  // Filter only fraudulent transactions (probability >= 0.3)
  const fraudulentTx = allTransactions.filter(tx => {
    const txTime = new Date(tx.timestamp).getTime()
    const hoursAgo24 = Date.now() - (24 * 60 * 60 * 1000)
    return tx.probability >= 0.3 && txTime >= hoursAgo24
  })

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

  if (fraudulentTx.length === 0) {
    return (
      <div className="card fraud-panel">
        <h3>Recent Fraud Detections (24h)</h3>
        <div className="empty-state">No fraudulent transactions detected in the last 24 hours</div>
      </div>
    )
  }

  return (
    <div className="card fraud-panel">
      <h3>Recent Fraud Detections (24h)</h3>
      <div className="fraud-stats">
        {fraudulentTx.length} fraud alert{fraudulentTx.length !== 1 ? 's' : ''} found
      </div>
      <div className="table-responsive fraud-scroll" ref={scrollContainerRef}>
        <table className="alert-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Risk Score</th>
              <th>Address</th>
              <th>Transaction</th>
            </tr>
          </thead>
          <tbody>
            {fraudulentTx.slice(0, displayCount).map((alert) => (
              <tr key={alert.tx_hash}>
                <td>{new Date(alert.timestamp).toLocaleString()}</td>
                <td>
                  <span className={`badge ${alert.probability > 0.7 ? 'bg-danger' : 'bg-warning'}`}>
                    {(alert.probability * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="font-mono">{alert.address.substring(0, 10)}...{alert.address.substring(38)}</td>
                <td>
                  <a
                    href={`https://etherscan.io/tx/0x${alert.tx_hash}`}
                    target="_blank"
                    rel="noreferrer"
                    className="etherscan-link"
                  >
                    View
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {displayCount < fraudulentTx.length && (
          <div className="load-more-indicator">
            Scroll for more... ({displayCount} of {fraudulentTx.length})
          </div>
        )}
      </div>
    </div>
  )
}

export default FraudAlerts
