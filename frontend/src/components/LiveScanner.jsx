import { useMemo, useState } from 'react'
import { Search, History, ArrowDownWideNarrow } from 'lucide-react'
import RiskBadge from './RiskBadge'
import Sparkline from './Sparkline'
import EmptyState from './EmptyState'
import { shortHash, shortAddr, formatEth, relativeTime } from '../lib/format'
import { riskColor, THREAT_THRESHOLD } from '../lib/risk'
import { indexByAddress, seriesFor } from '../lib/series'

const COLS = '74px minmax(0,1.35fr) minmax(0,1fr) 96px 64px 96px'
const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`
const ROW_CAP = 120 // rendered rows; the 24h window itself can hold thousands

const BANDS = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'threat', label: `≥${THREAT_THRESHOLD * 100}%`, test: (p) => p >= THREAT_THRESHOLD },
  { id: 'high', label: '≥80%', test: (p) => p >= 0.8 },
]

export default function LiveScanner({ transactions, selected, onSelect }) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('latest') // 'latest' | 'risk'
  const [band, setBand] = useState('all')

  const index = useMemo(() => indexByAddress(transactions), [transactions])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const pass = BANDS.find((b) => b.id === band).test
    const list = transactions.filter((t) => {
      if (!pass(t.probability || 0)) return false
      if (!q) return true
      return (
        t.tx_hash?.toLowerCase().includes(q) ||
        t.address?.toLowerCase().includes(q) ||
        t.to_address?.toLowerCase().includes(q)
      )
    })
    return sort === 'risk' ? list.sort((a, b) => (b.probability || 0) - (a.probability || 0)) : list
  }, [transactions, query, sort, band])

  const SortIcon = sort === 'latest' ? History : ArrowDownWideNarrow

  return (
    <div className="card elev-sm overflow-hidden rounded-md p-0">
      <div className="flex flex-wrap items-center gap-2.5 border-b border-line px-3.5 py-[11px]">
        <div className="text-sm" style={{ fontFamily: 'var(--font-heading)' }}>Scored transfers</div>
        <span className="mono text-[11px]" style={{ color: MUTED(45) }}>
          {rows.length.toLocaleString()} of {transactions.length.toLocaleString()}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="seg h-7">
            {BANDS.map((b) => (
              <label key={b.id} className="seg-opt">
                <input type="radio" name="rc-band" checked={band === b.id} onChange={() => setBand(b.id)} />
                {b.label}
              </label>
            ))}
          </div>
          <div className="relative">
            <Search size={13} className="pointer-events-none absolute left-[9px] top-1/2 -translate-y-1/2" style={{ color: MUTED(45) }} />
            <input
              className="input mono h-7 min-h-7 w-[200px] py-0 pl-[27px] pr-2.5 text-xs"
              style={{ background: 'var(--color-bg)' }}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="hash / address"
              aria-label="Filter by hash or address"
            />
          </div>
          <button
            className="btn btn-secondary h-7 px-2.5 text-xs"
            onClick={() => setSort((s) => (s === 'latest' ? 'risk' : 'latest'))}
          >
            <SortIcon size={13} />{sort === 'latest' ? 'Latest' : 'Risk'}
          </button>
        </div>
      </div>

      {/* One fixed grid for header + rows; the whole table scrolls sideways on narrow screens */}
      <div className="rc-scroll overflow-x-auto">
        <div className="min-w-[760px]">
          <div
            className="grid gap-2.5 border-b border-line px-3.5 py-2 text-[10.5px] uppercase tracking-[0.08em]"
            style={{ gridTemplateColumns: COLS, color: MUTED(50) }}
          >
            <div>Age</div><div>Tx / from</div><div>To</div>
            <div className="text-right">Value</div><div className="text-right">Trend</div><div className="text-right">Risk</div>
          </div>

          <div
            className="rc-scroll overflow-y-auto"
            style={{ maxHeight: selected ? 'max(300px, calc(100dvh - 620px))' : 'max(320px, calc(100dvh - 320px))' }}
          >
            {rows.slice(0, ROW_CAP).map((t, i) => {
              const color = riskColor(t.probability)
              const isSel = selected === t.tx_hash
              return (
                <div
                  key={t.tx_hash}
                  onClick={() => onSelect(t.tx_hash)}
                  className={`rc-row grid cursor-pointer items-center gap-2.5 border-b px-3.5 py-2 ${i === 0 && sort === 'latest' ? 'rc-flash' : ''}`}
                  style={{
                    gridTemplateColumns: COLS,
                    boxShadow: `inset 3px 0 0 ${color}`,
                    borderColor: MUTED(6),
                    background: isSel ? 'color-mix(in srgb, var(--color-accent) 10%, transparent)' : 'transparent',
                  }}
                >
                  <div className="mono text-[11px]" style={{ color: MUTED(45) }}>{relativeTime(t.timestamp)}</div>
                  <div className="min-w-0">
                    <div className="mono truncate text-xs text-accent">{shortHash(t.tx_hash, 16)}</div>
                    <div className="mono mt-px truncate text-[11px]" style={{ color: MUTED(55) }}>{shortAddr(t.address)}</div>
                  </div>
                  <div className="mono truncate text-xs" style={{ color: MUTED(80) }}>
                    {t.to_address ? shortAddr(t.to_address) : 'contract creation'}
                  </div>
                  <div className="mono text-right text-xs">{formatEth(t.value)} Ξ</div>
                  <div className="flex justify-end">
                    <Sparkline
                      values={seriesFor(index, t.address)}
                      w={56}
                      h={18}
                      color={color}
                      strokeWidth={1.3}
                    />
                  </div>
                  <div className="text-right"><RiskBadge probability={t.probability} /></div>
                </div>
              )
            })}

            {rows.length === 0 && (
              <EmptyState
                icon={Search}
                title={transactions.length ? 'Nothing matches that filter' : 'Listening for blocks'}
                description={
                  transactions.length
                    ? 'Try a different hash, address, or risk band.'
                    : 'Scored transfers appear here as the monitor writes them.'
                }
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
