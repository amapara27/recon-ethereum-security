import { useMemo, useState } from 'react'
import { ArrowLeft, ArrowUpRight, Eye, FileSearch, Copy, Check } from 'lucide-react'
import Sparkline from './Sparkline'
import { shortAddr, formatEth, formatPct, relativeTime, etherscanAddr } from '../lib/format'
import { riskColor, riskBand } from '../lib/risk'
import { addressTouches, counterparties } from '../lib/series'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`
const LABEL = 'text-[11px] uppercase tracking-[0.07em]'

// Everything here is read off the 24h alert window — no separate per-address endpoint exists,
// so the panel shows what the feed actually knows about the sender and says so.
export default function AddressPanel({ tx, alerts, onClear, onAudit, watched, onWatch }) {
  const address = tx.address
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(address)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* clipboard blocked */
    }
  }

  const { history, stats, peers } = useMemo(() => {
    const touches = addressTouches(alerts, address)
    const probs = touches.map((t) => t.probability || 0)
    const values = touches.map((t) => parseFloat(t.value) || 0)
    return {
      history: probs,
      peers: counterparties(alerts, address, 4),
      stats: [
        ['First seen', touches.length ? relativeTime(touches[0].timestamp) : '—'],
        ['Touches, 24h', touches.length.toLocaleString()],
        ['Mean value', `${(values.reduce((s, v) => s + v, 0) / (values.length || 1)).toFixed(4)} Ξ`],
        ['Peak risk', probs.length ? formatPct(Math.max(...probs)) : '—'],
      ],
    }
  }, [alerts, address])

  const color = riskColor(tx.probability)

  return (
    <div className="card elev-md rc-rise overflow-hidden rounded-md p-0">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-[11px]">
        <button className="btn btn-ghost px-1 py-0.5" onClick={onClear} aria-label="Close investigation">
          <ArrowLeft size={14} />
        </button>
        <span className={LABEL} style={{ color: MUTED(55) }}>Address investigation</span>
        <button className="btn btn-ghost ml-auto px-1.5 py-0.5 text-[11.5px]" onClick={() => onWatch(address)}>
          <Eye size={13} fill={watched ? 'currentColor' : 'none'} />{watched ? 'Tracked' : 'Track'}
        </button>
      </div>

      <div className="p-3.5">
        <div className="flex items-start gap-2">
          <div className="mono min-w-0 break-all text-[13px]">{address}</div>
          <button onClick={copy} aria-label="Copy address" className="btn btn-ghost mt-px shrink-0 px-1 py-0.5">
            {copied ? <Check size={13} style={{ color: 'var(--risk-safe)' }} /> : <Copy size={13} />}
          </button>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="mono text-[22px]" style={{ color }}>{formatPct(tx.probability)}</span>
          <span className="tag border" style={{ borderColor: color, color }}>{riskBand(tx.probability)}</span>
          <a
            href={etherscanAddr(address)}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-[11.5px] no-underline"
          >
            Etherscan<ArrowUpRight size={11} />
          </a>
        </div>

        <div className="mt-3">
          <Sparkline values={history} w={300} h={44} color={color} area full strokeWidth={1.6} opacity={1} />
        </div>
        <div className="text-[11px]" style={{ color: MUTED(45) }}>
          {history.length > 1
            ? `Score history · ${history.length} touches in the 24h window`
            : 'Only one scored touch in the window — no history to plot yet'}
        </div>

        <div className="mt-3.5 grid grid-cols-2 gap-[9px]">
          {stats.map(([k, v]) => (
            <div key={k} className="rounded-sm px-2.5 py-[9px]" style={{ background: 'var(--surface-2)' }}>
              <div className="text-[10.5px] uppercase tracking-[0.06em]" style={{ color: MUTED(50) }}>{k}</div>
              <div className="mono mt-[3px] text-[13px]">{v}</div>
            </div>
          ))}
        </div>

        <div className={`${LABEL} mb-2.5 mt-4`} style={{ color: MUTED(55) }}>Recent counterparties</div>
        {peers.length === 0 ? (
          <div className="text-[11.5px]" style={{ color: MUTED(45) }}>No other transfers with this address in the window.</div>
        ) : (
          <div className="flex flex-col gap-1.5">
            {peers.map((p) => (
              <div key={p.address} className="flex items-center gap-2 text-[11.5px]">
                <span className="mono" style={{ color: MUTED(75) }}>{shortAddr(p.address)}</span>
                <span className="mono ml-auto" style={{ color: riskColor(p.probability) }}>{formatPct(p.probability)}</span>
                <span className="mono w-14 text-right" style={{ color: MUTED(40) }}>{formatEth(p.value)} Ξ</span>
              </div>
            ))}
          </div>
        )}

        {tx.to_address && (
          <button className="btn btn-secondary btn-block mt-3.5 text-[12.5px]" onClick={() => onAudit(tx.to_address)}>
            <FileSearch size={14} />Audit counterparty contract
          </button>
        )}
      </div>
    </div>
  )
}
