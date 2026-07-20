import { useState } from 'react'
import { Copy, Check, ExternalLink } from 'lucide-react'
import { shortAddr, etherscanAddr } from '../lib/format'

// Truncated mono address with copy + Etherscan link. Used for from/to addresses.
export default function AddressPill({ address, label }) {
  const [copied, setCopied] = useState(false)
  if (!address) return <span className="font-mono text-xs text-muted">—</span>

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(address)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="font-mono text-xs text-ink">{label ? `${label} ` : ''}{shortAddr(address)}</span>
      <button
        onClick={copy}
        aria-label="Copy address"
        className="text-muted transition-colors hover:text-accent"
      >
        {copied ? <Check size={13} className="text-risk-safe" /> : <Copy size={13} />}
      </button>
      <a
        href={etherscanAddr(address)}
        target="_blank"
        rel="noreferrer"
        aria-label="View on Etherscan"
        className="text-muted transition-colors hover:text-accent"
      >
        <ExternalLink size={13} />
      </a>
    </span>
  )
}
