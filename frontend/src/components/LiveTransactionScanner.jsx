function LiveTransactionScanner({ transactions }) {

  if (transactions.length === 0) {
    return <div className="empty-state">Waiting for transactions...</div>
  }

  return (
    <div className="card scanner-panel">
      <h3>Live Transaction Scanner</h3>
      <div className="scanner-info">Showing last 25 scanned transactions</div>
      <div className="table-responsive">
        <table className="scanner-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Status</th>
              <th>Address</th>
              <th>Risk Score</th>
              <th>Transaction</th>
            </tr>
          </thead>
          <tbody>
            {transactions.slice(0, 25).map((tx) => {
              const isFraud = tx.probability >= 0.3;
              return (
                <tr key={tx.tx_hash} className={isFraud ? 'fraud-row' : 'clean-row'}>
                  <td>{new Date(tx.timestamp).toLocaleTimeString()}</td>
                  <td>
                    <span className={`status-badge ${isFraud ? 'status-fraud' : 'status-clean'}`}>
                      {isFraud ? 'FRAUD' : 'CLEAN'}
                    </span>
                  </td>
                  <td className="font-mono">{tx.address.substring(0, 10)}...{tx.address.substring(38)}</td>
                  <td>
                    <span className={`badge ${tx.probability > 0.7 ? 'bg-danger' : tx.probability >= 0.3 ? 'bg-warning' : 'bg-safe'}`}>
                      {(tx.probability * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td>
                    <a
                      href={`https://etherscan.io/tx/0x${tx.tx_hash}`}
                      target="_blank"
                      rel="noreferrer"
                      className="etherscan-link"
                    >
                      View
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default LiveTransactionScanner
