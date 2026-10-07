import { useMemo, useState } from 'react'
import { X } from 'lucide-react'
import RangeBar from './RangeBar'
import EmptyState from './EmptyState'
import { shortAddr, formatPct, relativeTime, etherscanAddr } from '../lib/format'
import { riskColor, THREAT_THRESHOLD } from '../lib/risk'
import { addressTouches } from '../lib/series'
import { ADDRESS_RE } from '../hooks/useWatchlist'

export default function Watchlist({ watched, alerts, onAdd, onRemove }) {
  const [query, setQuery] = useState('')
  const valid = ADDRESS_RE.test(query.trim())

  // Everything below is derived from the same 24h window the scanner uses.
  const rows = useMemo(
    () =>
      watched.map((address) => {
        const touches = addressTouches(alerts, address)
        const probs = touches.map((t) => t.probability || 0)
        const last = touches[touches.length - 1]
        return {
          address,
          seen: touches.length > 0,
          history: probs,
          probability: probs.length ? probs[probs.length - 1] : null,
          drift: probs.length > 1 ? probs[probs.length - 1] - probs[0] : null,
          touches: touches.length,
          lastSeen: last ? relativeTime(last.timestamp) : '—',
        }
      }),
    [watched, alerts],
  )

  const submit = (e) => {
    e.preventDefault()
    if (onAdd(query)) setQuery('')
  }

  return (
    <div className="flex flex-col gap-4">
      <form className="sheet px-4 py-4 sm:px-5" onSubmit={submit}>
        <label htmlFor="watch-address" className="th">Address to track</label>
        <div className="mt-1.5 flex flex-col gap-2 sm:flex-row">
          <input
            id="watch-address"
            className="input mono min-h-[40px] flex-1 text-[14px]"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="0x…"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="btn btn-primary min-h-[40px] px-5" disabled={!valid}>Track</button>
        </div>
        <p className="mt-2.5 max-w-[78ch] text-[12.5px] text-ink-3">Saved in this browser only.</p>
      </form>

      <section className="sheet" aria-label="Tracked addresses">
        <div className="sheet-head">
          <h2 className="sheet-title">Tracked</h2>
          <span className="mono text-[12.5px] text-ink-3">{rows.length}</span>
        </div>

        <div className="watch-grid hidden border-b border-line px-4 py-2 md:grid">
          <div className="th">Address</div>
          <div className="th">Range · ref &lt; {THREAT_THRESHOLD * 100}%</div>
          <div className="th text-right">Latest</div>
          <div className="th text-right">24h drift</div>
          <div className="th text-right">Touches</div>
          <div className="th text-right">Last seen</div>
          <div />
        </div>

        {rows.map((w) => {
          const color = w.seen ? riskColor(w.probability) : 'var(--ink-3)'
          const driftColor =
            w.drift == null ? 'var(--ink-3)'
              : w.drift > 0.05 ? 'var(--risk-high)'
              : 'var(--ink-2)'
          return (
            <div key={w.address} className="watch-grid items-center border-b border-line px-4 py-3">
              <div className="min-w-0" style={{ gridArea: 'addr' }}>
                <a href={etherscanAddr(w.address)} target="_blank" rel="noreferrer" className="mono block truncate text-[13.5px] font-semibold text-ink no-underline hover:underline">
                  {shortAddr(w.address)}
                </a>
                <div className="text-[12px] text-ink-3">{w.seen ? 'in the live window' : 'not seen in the last 24h'}</div>
              </div>
              <div style={{ gridArea: 'bar' }}>
                {w.seen ? <RangeBar value={w.probability} history={w.history} /> : <span className="text-[12.5px] text-ink-3">No score yet</span>}
              </div>
              <div className={`mono text-right text-[14px] ${w.probability >= THREAT_THRESHOLD ? 'font-bold' : ''}`} style={{ gridArea: 'pct', color }}>
                {w.probability == null ? '—' : formatPct(w.probability)}
              </div>
              <div className="text-[13px] md:text-right" style={{ gridArea: 'drift', color: driftColor }}>
                {w.drift == null ? '—' : `${w.drift >= 0 ? '+' : ''}${(w.drift * 100).toFixed(0)} pts`}
              </div>
              <div className="text-right text-[13px] text-ink-2" style={{ gridArea: 'touch' }}>
                <span className="md:hidden">touches </span>{w.touches}
              </div>
              <div className="text-right text-[12.5px] text-ink-3" style={{ gridArea: 'seen' }}>{w.lastSeen}</div>
              <button
                className="btn btn-ghost justify-self-end"
                style={{ gridArea: 'rm' }}
                onClick={() => onRemove(w.address)}
                aria-label={`Stop tracking ${shortAddr(w.address)}`}
              >
                <X size={15} />
              </button>
            </div>
          )
        })}

        {rows.length === 0 && (
          <EmptyState title="Nothing tracked yet" description="Paste an address above, or open a result in the scanner and press Track." />
        )}
      </section>
    </div>
  )
}
