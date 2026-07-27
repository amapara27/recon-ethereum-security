// Shared formatting helpers for on-chain data.

// tx_hash from the backend has NO 0x prefix; add it for display + Etherscan links.
const hx = (h) => (h?.startsWith('0x') ? h : `0x${h || ''}`)

export const shortHash = (h, n = 10) => (h ? `${hx(h).slice(0, n + 2)}…` : '—')

export const shortAddr = (a) =>
  a ? `${hx(a).slice(0, 6)}…${a.slice(-4)}` : '—'

export const etherscanTx = (h) => `https://etherscan.io/tx/${hx(h)}`
export const etherscanAddr = (a) => `https://etherscan.io/address/${hx(a)}`

export const formatEth = (v) => {
  const n = parseFloat(v)
  return Number.isFinite(n) ? n.toFixed(4) : '0.0000'
}

export const formatPct = (p) => `${((p || 0) * 100).toFixed(1)}%`

// Epoch milliseconds from either a second/millisecond epoch or an ISO string. NaN if unparseable.
export function toMs(timestamp) {
  if (!timestamp) return NaN
  if (typeof timestamp === 'number') return timestamp < 1e12 ? timestamp * 1000 : timestamp
  return new Date(timestamp).getTime()
}

// Relative time; handles both second and millisecond epochs plus ISO strings.
export function relativeTime(timestamp) {
  if (!timestamp) return 'just now'
  const now = Date.now()
  const time = toMs(timestamp)
  if (Number.isNaN(time) || time > now) return 'just now'
  const s = Math.floor((now - time) / 1000)
  if (s < 5) return 'just now'
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`
  return `${Math.floor(h / 24)}d ago`
}
