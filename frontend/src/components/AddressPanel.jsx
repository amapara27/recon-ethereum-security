import { useEffect, useMemo, useState } from 'react'
import { X, ArrowUpRight, Eye, FileSearch, Copy, Check } from 'lucide-react'
import RangeBar from './RangeBar'
import { shortAddr, formatEth, formatPct, relativeTime, etherscanAddr } from '../lib/format'
import { riskColor, riskBand } from '../lib/risk'
import { addressTouches, counterparties } from '../lib/series'

// Everything here is read off the 24h alert window — no separate per-address endpoint exists,
// so the panel shows what the feed actually knows about the sender and says so.
// Below xl it covers the screen as its own sheet; from xl it sits in the right column.
export default function AddressPanel({ tx, alerts, onClear, onAudit, watched, onWatch }) {
  const address = tx.address
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClear()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClear])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(address)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      /* clipboard blocked */
    }
  }

  const { history, stats, peers, recent } = useMemo(() => {
    const touches = addressTouches(alerts, address)
    const probs = touches.map((t) => t.probability || 0)
    const values = touches.map((t) => parseFloat(t.value) || 0)
    return {
      history: probs,
      recent: touches.slice(-6).reverse(),
      peers: counterparties(alerts, address, 4),
      stats: [
        ['First seen', touches.length ? relativeTime(touches[0].timestamp) : '—', false],
        ['Touches, 24h', touches.length.toLocaleString()],
        ['Mean value', `${(values.reduce((s, v) => s + v, 0) / (values.length || 1)).toFixed(4)} Ξ`],
        ['Peak score', probs.length ? formatPct(Math.max(...probs)) : '—'],
      ],
    }
  }, [alerts, address])

  const color = riskColor(tx.probability)

  return (
    <section
      className="sheet fixed inset-0 z-50 overflow-auto border-0 xl:static xl:z-auto xl:overflow-visible xl:border"
      aria-label="Address result"
    >
      <div className="sheet-head sticky top-0 bg-sheet xl:static">
        <h2 className="sheet-title">Sender</h2>
        <a href={etherscanAddr(address)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12.5px] font-semibold no-underline">
          Etherscan<ArrowUpRight size={12} />
        </a>
        <div className="ml-auto flex items-center gap-1">
          <button className="btn btn-ghost text-[12.5px]" onClick={() => onWatch(address)} aria-pressed={watched}>
            <Eye size={14} fill={watched ? 'currentColor' : 'none'} />{watched ? 'Tracked' : 'Track'}
          </button>
          <button className="btn btn-ghost" onClick={onClear} aria-label="Close result">
            <X size={16} />
          </button>
        </div>
      </div>

      <div className="px-4 pb-5 pt-3.5">
        <div className="flex items-start gap-2">
          <div className="mono min-w-0 break-all text-[13.5px]">{address}</div>
          <button onClick={copy} aria-label="Copy address" className="btn btn-ghost -mt-1 shrink-0">
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        </div>

        <div className="mt-4 flex items-baseline gap-3">
          <span className="mono text-[30px] font-semibold leading-none" style={{ color }}>{formatPct(tx.probability)}</span>
          <span className="text-[13.5px] font-bold" style={{ color }}>{riskBand(tx.probability)}</span>
          <span className="ml-auto text-[12.5px] text-ink-3">this transfer, {relativeTime(tx.timestamp)}</span>
        </div>
        <RangeBar value={tx.probability} history={history} height={22} axis className="mt-3" />
        <p className="mt-1 text-[12.5px] text-ink-3">
          {history.length > 1
            ? `Hollow dots: the ${history.length - 1} other scores for this address in the window.`
            : 'Only one scored touch in the window, so there is no history to compare.'}
        </p>

        <dl className="m-0 mt-4 grid grid-cols-2 border-t border-line">
          {stats.map(([k, v, mono = true]) => (
            <div key={k} className="border-b border-line py-2 odd:pr-3 even:border-l even:pl-3">
              <dt className="th">{k}</dt>
              <dd className={`m-0 mt-0.5 text-[14px] ${mono ? 'mono' : ''}`}>{v}</dd>
            </div>
          ))}
        </dl>

        {recent.length > 1 && (
          <>
            <h3 className="th mt-5">Previous results</h3>
            <table className="mt-1.5 w-full border-collapse text-[12.5px]">
              <tbody>
                {recent.map((r) => (
                  <tr key={r.tx_hash} className="border-b border-line">
                    <td className="py-1.5 text-ink-3">{relativeTime(r.timestamp)}</td>
                    <td className="mono py-1.5 text-right text-ink-2">{formatEth(r.value)} Ξ</td>
                    <td className="mono py-1.5 text-right font-semibold" style={{ color: riskColor(r.probability) }}>{formatPct(r.probability)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        <h3 className="th mt-5">Recent counterparties</h3>
        {peers.length === 0 ? (
          <p className="mt-1.5 text-[12.5px] text-ink-3">No other transfers with this address in the window.</p>
        ) : (
          <table className="mt-1.5 w-full border-collapse text-[12.5px]">
            <tbody>
              {peers.map((p) => (
                <tr key={p.address} className="border-b border-line">
                  <td className="mono py-1.5 text-ink-2">{shortAddr(p.address)}</td>
                  <td className="mono py-1.5 text-right text-ink-3">{formatEth(p.value)} Ξ</td>
                  <td className="mono py-1.5 text-right font-semibold" style={{ color: riskColor(p.probability) }}>{formatPct(p.probability)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {tx.to_address && (
          <>
            <button className="btn btn-secondary mt-5 w-full" onClick={() => onAudit(tx.to_address)}>
              <FileSearch size={15} />Audit the recipient contract
            </button>
          </>
        )}
      </div>
    </section>
  )
}
