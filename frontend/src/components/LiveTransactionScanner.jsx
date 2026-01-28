import { useState, useMemo } from 'react'
import CyberCard from './CyberCard'

// --- ICONS ---
const SearchIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
)

const ScannerIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
  </svg>
)

// --- HELPERS ---
function getRelativeTime(timestamp) {
  if (!timestamp) return 'Just now'
  const now = Date.now()
  let time = typeof timestamp === 'number'
    ? (timestamp < 1e12 ? timestamp * 1000 : timestamp)
    : new Date(timestamp).getTime()

  if (isNaN(time) || time > now) return 'Just now'
  const diff = now - time
  const seconds = Math.floor(diff / 1000)

  if (seconds < 5) return 'Just now'
  if (seconds < 60) return `${seconds} seconds ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`
  return `${Math.floor(minutes / 60)}h ago`
}

// --- COMPONENT ---
function LiveTransactionScanner({ transactions }) {
  const [searchQuery, setSearchQuery] = useState('')

  const filteredTransactions = useMemo(() => {
    if (!searchQuery.trim()) return transactions.slice(0, 50)
    const query = searchQuery.toLowerCase().trim()
    return transactions.filter(tx =>
      tx.tx_hash?.toLowerCase().includes(query) ||
      tx.address?.toLowerCase().includes(query) ||
      tx.to_address?.toLowerCase().includes(query)
    ).slice(0, 50)
  }, [transactions, searchQuery])

  const getRiskColor = (prob) => {
    if (prob >= 0.8) return 'var(--accent-danger)'
    if (prob >= 0.5) return 'var(--warning)'
    return 'var(--accent-success)'
  }

  return (
    <CyberCard title="NETWORK TRAFFIC" icon={<ScannerIcon />} style={{ height: '100%' }}>
      {/* Search Bar */}
      <div style={{ marginBottom: '16px', position: 'relative' }}>
        <input
          type="text"
          placeholder="FILTER HASH / ADDRESS..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            width: '100%',
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid var(--border-subtle)',
            padding: '10px 12px 10px 36px',
            color: 'white',
            fontFamily: 'var(--font-mono)',
            borderRadius: '6px',
            outline: 'none',
            fontSize: '13px'
          }}
        />
        <div style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-tertiary)' }}>
          <SearchIcon />
        </div>
      </div>

      {/* List Header */}
      <div style={{ display: 'grid', gridTemplateColumns: '120px 1.5fr 1fr 100px 100px', padding: '0 8px 8px', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-tertiary)', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
        <div>ELAPSED</div>
        <div>TRANSACTION / SOURCE</div>
        <div>DESTINATION</div>
        <div style={{ textAlign: 'right' }}>VALUE</div>
        <div style={{ textAlign: 'right' }}>RISK SCORE</div>
      </div>

      {/* List Itmes */}
      <div className="scanner-list" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {filteredTransactions.map((tx, index) => {
          const riskColor = getRiskColor(tx.probability || 0);
          // Only flash the first item if it's "new" (less than 5 seconds old or simply the 0th index of a fresh list)
          const isNew = index === 0 && (!tx.timestamp || (Date.now() - (typeof tx.timestamp === 'number' ? (tx.timestamp < 1e12 ? tx.timestamp * 1000 : tx.timestamp) : new Date(tx.timestamp).getTime())) < 5000);

          return (
            <div key={tx.tx_hash}
              className={isNew ? 'new-item-flash' : ''}
              style={{
                display: 'grid',
                gridTemplateColumns: '120px 1.5fr 1fr 100px 100px',
                padding: '12px 8px',
                background: 'rgba(255,255,255,0.02)',
                borderLeft: `2px solid ${riskColor}`,
                borderRadius: '0 4px 4px 0',
                alignItems: 'center',
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                transition: 'all 0.3s ease'
              }}
            >
              <div style={{
                color: isNew ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontWeight: isNew ? 'bold' : 'normal',
                fontSize: '11px'
              }}>
                {getRelativeTime(tx.timestamp)}
              </div>

              <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingRight: '12px' }}>
                <div style={{ color: 'var(--accent-cyan)', marginBottom: '2px' }}>
                  {tx.tx_hash ? `${tx.tx_hash.slice(0, 10)}...` : 'Processing...'}
                </div>
                <div style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
                  {tx.address ? `${tx.address.slice(0, 6)}...${tx.address.slice(-4)}` : 'UNKNOWN'}
                </div>
              </div>

              <div style={{ color: 'var(--text-secondary)', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {tx.to_address ? `${tx.to_address.slice(0, 6)}...${tx.to_address.slice(-4)}` : 'CONTRACT_VOID'}
              </div>

              <div style={{ textAlign: 'right', color: 'var(--accent-purple)', fontWeight: 'bold', fontSize: '11px' }}>
                {tx.value ? parseFloat(tx.value).toFixed(4) : '0.0000'} ETH
              </div>

              <div style={{ textAlign: 'right', fontWeight: 'bold', color: riskColor }}>
                {((tx.probability || 0) * 100).toFixed(1)}%
              </div>
            </div>
          )
        })}

        {transactions.length === 0 && (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                 // LISTENING FOR BLOCKS...
            <br />
            <span style={{ fontSize: '24px', display: 'block', marginTop: '12px', animation: 'pulse 1.5s infinite' }}>●</span>
          </div>
        )}
      </div>
    </CyberCard>
  )
}

export default LiveTransactionScanner
