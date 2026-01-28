import CyberCard from './CyberCard'

const AlertIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
)

function FraudAlerts({ allTransactions }) {
  // Filter for high risk (prob > 50%)
  const alerts = allTransactions.filter(tx => (tx.probability || 0) > 0.5)

  if (alerts.length === 0) {
    return (
      <CyberCard title="ACTIVE THREATS" icon={<AlertIcon />} variant="default">
        <div style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--accent-success)',
          opacity: 0.7,
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🛡️</div>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '18px' }}>SYSTEM SECURE</div>
          <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '8px' }}>NO ACTIVE THREATS DETECTED</div>
        </div>
      </CyberCard>
    )
  }

  return (
    <CyberCard title={`THREATS DETECTED (${alerts.length})`} icon={<AlertIcon />} variant="danger">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {alerts.map(alert => (
          <div key={alert.tx_hash} style={{
            background: 'rgba(255, 42, 81, 0.05)',
            border: '1px solid var(--accent-danger)',
            borderRadius: '8px',
            padding: '16px',
            position: 'relative'
          }}>
            {/* Glowing dot */}
            <div style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: 'var(--accent-danger)',
              boxShadow: '0 0 10px var(--accent-danger)'
            }} />

            <div style={{ color: 'var(--accent-danger)', fontFamily: 'var(--font-display)', fontSize: '14px', marginBottom: '8px' }}>
              HIGH RISK TRANSACTION
            </div>

            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
              HASH: <span style={{ color: 'white' }}>{alert.tx_hash?.slice(0, 12)}...</span>
            </div>

            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-secondary)' }}>
              SCORE: <span style={{ color: 'var(--accent-danger)', fontWeight: 'bold' }}>{((alert.probability || 0) * 100).toFixed(2)}%</span>
            </div>

            <a
              href={`https://etherscan.io/tx/0x${alert.tx_hash}`}
              target="_blank"
              rel="noreferrer"
              style={{
                display: 'inline-block',
                marginTop: '12px',
                fontSize: '11px',
                color: 'var(--accent-danger)',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                borderBottom: '1px dotted var(--accent-danger)'
              }}
            >
              Investigate on Etherscan &rarr;
            </a>
          </div>
        ))}
      </div>
    </CyberCard>
  )
}

export default FraudAlerts
