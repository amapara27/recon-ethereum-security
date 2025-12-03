function DisplayAlerts({ alerts }) {
  
  if (alerts.length === 0) {
    return <div className="empty-state">No alerts found (or waiting for data)...</div>
  }

  return (
    <div className="card">
      <h3>🚨 Live Fraud Feed</h3>
      <div className="table-responsive">
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
            {alerts.map((alert) => (
              <tr key={alert.tx_hash}>
                <td>{new Date(alert.timestamp).toLocaleTimeString()}</td>
                <td>
                  {/* Conditional styling for risk score */}
                  <span className={`badge ${alert.probability > 0.7 ? 'bg-danger' : 'bg-warning'}`}>
                    {(alert.probability * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="font-mono">{alert.address}</td>
                <td>
                  <a 
                    href={`https://etherscan.io/tx/0x${alert.tx_hash}`} 
                    target="_blank" 
                    rel="noreferrer"
                    className="etherscan-link"
                  >
                    View ↗
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default DisplayAlerts