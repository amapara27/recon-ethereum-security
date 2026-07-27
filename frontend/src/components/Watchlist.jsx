import { useMemo, useState } from 'react'
import { Plus, X, Eye } from 'lucide-react'
import Sparkline from './Sparkline'
import EmptyState from './EmptyState'
import { shortAddr, formatPct, relativeTime } from '../lib/format'
import { riskColor, THREAT_THRESHOLD } from '../lib/risk'
import { addressTouches } from '../lib/series'
import { ADDRESS_RE } from '../hooks/useWatchlist'

const COLS = 'minmax(0,1.2fr) 84px 110px 90px 120px 34px'
const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`

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
    <div className="mx-auto flex max-w-[1000px] flex-col gap-3.5">
      <form className="card elev-sm rounded-md p-4" onSubmit={submit}>
        <div className="flex flex-wrap gap-[9px]">
          <input
            className="input mono min-h-[38px] flex-1 text-[13px] sm:min-w-[260px]"
            style={{ background: 'var(--color-bg)' }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="0x… address to track"
            aria-label="Address to track"
          />
          <button type="submit" className="btn btn-primary min-h-[38px] px-4" disabled={!valid}>
            <Plus size={14} />Track
          </button>
        </div>
        <div className="mt-2.5 text-xs" style={{ color: MUTED(55) }}>
          Tracked addresses are scored from the live feed — score, drift and last-seen come from the
          last 24 hours of transfers. Anything at or above {THREAT_THRESHOLD * 100}% also shows in Active threats.
        </div>
      </form>

      <div className="card elev-sm overflow-hidden rounded-md p-0">
        <div className="rc-scroll overflow-x-auto">
          <div className="min-w-[720px]">
            <div
              className="grid gap-2.5 border-b border-line px-[15px] py-[9px] text-[10.5px] uppercase tracking-[0.08em]"
              style={{ gridTemplateColumns: COLS, color: MUTED(50) }}
            >
              <div>Address</div><div>Score</div><div>24h drift</div>
              <div className="text-right">Touches</div><div className="text-right">Last seen</div><div />
            </div>

            {rows.map((w) => {
              const color = w.seen ? riskColor(w.probability) : MUTED(35)
              const driftColor =
                w.drift == null ? MUTED(50)
                  : w.drift > 0.05 ? 'var(--risk-high)'
                  : w.drift < -0.02 ? 'var(--risk-safe)'
                  : MUTED(50)
              return (
                <div
                  key={w.address}
                  className="grid items-center gap-2.5 border-b px-[15px] py-3"
                  style={{ gridTemplateColumns: COLS, boxShadow: `inset 3px 0 0 ${color}`, borderColor: MUTED(6) }}
                >
                  <div className="min-w-0">
                    <div className="mono truncate text-[12.5px]">{shortAddr(w.address)}</div>
                    <div className="mt-0.5 text-[11px]" style={{ color: MUTED(50) }}>
                      {w.seen ? 'seen in the live window' : 'not seen in the last 24h'}
                    </div>
                  </div>
                  <div className="mono text-[13px]" style={{ color }}>
                    {w.probability == null ? '—' : formatPct(w.probability)}
                  </div>
                  <div className="flex items-center gap-[7px]">
                    <Sparkline values={w.history} w={46} h={16} color={color} strokeWidth={1.3} />
                    <span className="mono text-[11.5px]" style={{ color: driftColor }}>
                      {w.drift == null ? '—' : `${w.drift >= 0 ? '+' : ''}${(w.drift * 100).toFixed(0)}pp`}
                    </span>
                  </div>
                  <div className="mono text-right text-xs" style={{ color: MUTED(65) }}>{w.touches}</div>
                  <div className="mono text-right text-[11.5px]" style={{ color: MUTED(45) }}>{w.lastSeen}</div>
                  <button
                    className="btn btn-ghost justify-self-end px-1 py-0.5"
                    onClick={() => onRemove(w.address)}
                    aria-label={`Stop tracking ${shortAddr(w.address)}`}
                  >
                    <X size={13} />
                  </button>
                </div>
              )
            })}

            {rows.length === 0 && (
              <EmptyState
                icon={Eye}
                title="No addresses tracked yet"
                description="Paste an address above, or pin one from the scanner."
                className="px-5 py-[52px]"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
