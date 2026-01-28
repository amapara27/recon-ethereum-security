import { useState } from 'react'
import CyberCard from './CyberCard'

const WalletIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="5" width="20" height="14" rx="2" />
    <line x1="2" y1="10" x2="22" y2="10" />
  </svg>
)

const AddIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
)

function WalletWatcher() {
  const [address, setAddress] = useState('')
  const [watchedList, setWatchedList] = useState([
    { address: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', label: 'Whale 1', status: 'Active' },
    { address: '0x123...456', label: 'Suspicious Setup', status: 'Flagged' }
  ])

  const handleAdd = (e) => {
    e.preventDefault()
    if (!address) return
    setWatchedList([...watchedList, { address, label: 'New Watch', status: 'Pending' }])
    setAddress('')
  }

  return (
    <CyberCard title="WALLET SURVEILLANCE" icon={<WalletIcon />}>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', height: '100%' }}>
        {/* Left Column: Input and Stats */}
        <div>
          <form onSubmit={handleAdd} style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="Track Wallet Address (0x...)"
              style={{
                flex: 1,
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-subtle)',
                padding: '12px 16px',
                borderRadius: '8px',
                color: 'white',
                fontFamily: 'var(--font-mono)'
              }}
            />
            <button
              type="submit"
              style={{
                background: 'var(--accent-cyan-dim)', // Use dim for button bg
                border: '1px solid var(--accent-cyan)',
                color: 'var(--accent-cyan)',
                padding: '0 20px',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontWeight: '600'
              }}
            >
              <AddIcon /> TRACK
            </button>
          </form>

          <div className="watched-list">
            <h4 style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '12px', fontFamily: 'var(--font-mono)' }}>ACTIVE TARGETS</h4>
            {watchedList.map((item, idx) => (
              <div key={idx} style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px',
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--border-subtle)',
                marginBottom: '8px',
                borderRadius: '8px'
              }}>
                <div>
                  <div style={{ color: 'white', fontWeight: '500', fontSize: '14px' }}>{item.label}</div>
                  <div style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{item.address.slice(0, 16)}...</div>
                </div>
                <span style={{
                  fontSize: '11px',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  background: item.status === 'Flagged' ? 'var(--accent-danger-dim)' : 'var(--accent-success-dim)',
                  color: item.status === 'Flagged' ? 'var(--accent-danger)' : 'var(--accent-success)',
                  border: `1px solid ${item.status === 'Flagged' ? 'var(--accent-danger)' : 'var(--accent-success)'}`
                }}>
                  {item.status.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Mini Chart or Info */}
        <div style={{
          background: 'rgba(0,0,0,0.2)',
          borderRadius: '8px',
          border: '1px dashed var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          color: 'var(--text-tertiary)'
        }}>
          <div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.3 }}>📊</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>ACTIVITY GRAPH</div>
          <div style={{ fontSize: '10px', marginTop: '4px' }}>No Data Available</div>
        </div>
      </div>
    </CyberCard>
  )
}

export default WalletWatcher
