import { useState, useMemo } from 'react'

// SVG Icons
function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

function ClearIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

function ArrowRightIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  )
}

function ExternalLinkIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  )
}

function EthIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M12 1.5L5.5 12.25L12 16.5L18.5 12.25L12 1.5Z" fill="currentColor" opacity="0.6" />
      <path d="M12 16.5L5.5 12.25L12 22.5L18.5 12.25L12 16.5Z" fill="currentColor" />
    </svg>
  )
}

function ScannerIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <line x1="7" y1="12" x2="17" y2="12" />
    </svg>
  )
}

// Generate gradient avatar based on address hash
function AddressAvatar({ address, size = 32 }) {
  // Create a simple hash from the address for consistent colors
  const hash = address.slice(2, 10)
  const hue1 = parseInt(hash.slice(0, 4), 16) % 360
  const hue2 = (hue1 + 40) % 360

  const gradientId = `avatar-${address.slice(2, 10)}`

  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={`hsl(${hue1}, 70%, 60%)`} />
          <stop offset="100%" stopColor={`hsl(${hue2}, 70%, 50%)`} />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="16" fill={`url(#${gradientId})`} />
      <text
        x="16"
        y="16"
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize="10"
        fontWeight="600"
        fontFamily="var(--font-mono)"
      >
        {address.slice(2, 4).toUpperCase()}
      </text>
    </svg>
  )
}

// Format relative time
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

function LiveTransactionScanner({ transactions }) {
  const [searchQuery, setSearchQuery] = useState('')

  // Filter transactions based on search query
  const filteredTransactions = useMemo(() => {
    if (!searchQuery.trim()) return transactions.slice(0, 25)

    const query = searchQuery.toLowerCase().trim()
    return transactions.filter(tx =>
      tx.tx_hash?.toLowerCase().includes(query) ||
      tx.address?.toLowerCase().includes(query) ||
      tx.to_address?.toLowerCase().includes(query)
    ).slice(0, 25)
  }, [transactions, searchQuery])

  const handleClear = () => {
    setSearchQuery('')
  }

  // Get risk assessment
  const getRiskLevel = (probability) => {
    if (probability >= 0.7) return { level: 'Fraudulent', class: 'risk-fraudulent' }
    if (probability >= 0.3) return { level: 'Suspicious', class: 'risk-suspicious' }
    return { level: 'Safe', class: 'risk-safe' }
  }

  if (transactions.length === 0) {
    return (
      <div className="card scanner-panel">
        <h3>Live Transaction Scanner</h3>
        <div className="scanner-empty-state">
          <ScannerIcon />
          <div className="scanner-empty-title">Scanning Network</div>
          <div className="scanner-empty-text">Waiting for transactions to appear...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="card scanner-panel">
      <h3>Live Transaction Scanner</h3>

      {/* Search Input */}
      <div className="scanner-search-container">
        <div className="scanner-search-wrapper">
          <span className="scanner-search-icon">
            <SearchIcon />
          </span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Enter transaction hash or wallet address"
            className="scanner-search-input"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={handleClear}
              className="scanner-search-clear"
              title="Clear search"
            >
              <ClearIcon />
            </button>
          )}
        </div>
      </div>

      <div className="scanner-info">
        {searchQuery
          ? `${filteredTransactions.length} result${filteredTransactions.length !== 1 ? 's' : ''} found`
          : 'Showing last 25 scanned transactions'
        }
      </div>

      {/* Transaction List */}
      <div className="scanner-transactions-scroll">
        <div className="scanner-transactions-list">
          {filteredTransactions.map((tx) => {
            const risk = getRiskLevel(tx.probability)
            const fromAddress = tx.address || '0x0000000000000000000000000000000000000000'
            const toAddress = tx.to_address || '0x0000000000000000000000000000000000000000'

            return (
              <div key={tx.tx_hash} className={`scanner-tx-card ${risk.class}`}>
                {/* Risk Badge */}
                <div className="scanner-tx-risk">
                  <span className={`scanner-risk-badge ${risk.class}`}>
                    {risk.level}
                  </span>
                  <span className="scanner-risk-score">
                    {(tx.probability * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Address Flow */}
                <div className="scanner-tx-flow">
                  <div className="scanner-address from">
                    <AddressAvatar address={fromAddress} size={36} />
                    <div className="scanner-address-info">
                      <span className="scanner-address-label">From</span>
                      <span className="scanner-address-hash font-mono">
                        {fromAddress.slice(0, 6)}...{fromAddress.slice(-4)}
                      </span>
                    </div>
                  </div>

                  <div className="scanner-flow-arrow">
                    <ArrowRightIcon />
                  </div>

                  <div className="scanner-address to">
                    <AddressAvatar address={toAddress} size={36} />
                    <div className="scanner-address-info">
                      <span className="scanner-address-label">To</span>
                      <span className="scanner-address-hash font-mono">
                        {toAddress.slice(0, 6)}...{toAddress.slice(-4)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Transaction Details */}
                <div className="scanner-tx-details">
                  {/* Value */}
                  <div className="scanner-tx-value">
                    <EthIcon />
                    <span className="scanner-value-amount">
                      {tx.value ? parseFloat(tx.value).toFixed(4) : '0.0000'}
                    </span>
                    <span className="scanner-value-label">ETH</span>
                  </div>

                  {/* Gas Progress */}
                  {tx.gas_used && tx.gas_limit && (
                    <div className="scanner-gas-info">
                      <div className="scanner-gas-header">
                        <span className="scanner-gas-label">Gas</span>
                        <span className="scanner-gas-value">
                          {((tx.gas_used / tx.gas_limit) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="scanner-gas-bar">
                        <div
                          className="scanner-gas-fill"
                          style={{ width: `${Math.min((tx.gas_used / tx.gas_limit) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Footer: Meta + Actions */}
                <div className="scanner-tx-footer">
                  <div className="scanner-tx-meta">
                    <span className="scanner-tx-time">{getRelativeTime(tx.timestamp)}</span>
                    {tx.block_number && (
                      <>
                        <span className="scanner-meta-divider">•</span>
                        <span className="scanner-tx-block">Block {tx.block_number.toLocaleString()}</span>
                      </>
                    )}
                  </div>

                  <a
                    href={`https://etherscan.io/tx/0x${tx.tx_hash}`}
                    target="_blank"
                    rel="noreferrer"
                    className="scanner-etherscan-btn"
                  >
                    <span>View on Etherscan</span>
                    <ExternalLinkIcon />
                  </a>
                </div>
              </div>
            )
          })}
        </div>

        {filteredTransactions.length === 0 && searchQuery && (
          <div className="scanner-no-results">
            <p>No transactions found matching "{searchQuery}"</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default LiveTransactionScanner
