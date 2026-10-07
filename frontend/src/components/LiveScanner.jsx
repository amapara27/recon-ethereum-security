import { useMemo, useState } from 'react'
import { Search, ArrowDown } from 'lucide-react'
import RangeBar from './RangeBar'
import EmptyState from './EmptyState'
import { shortHash, shortAddr, formatEth, formatPct, relativeTime } from '../lib/format'
import { riskColor, riskFlag, THREAT_THRESHOLD } from '../lib/risk'
import { indexByAddress, seriesFor } from '../lib/series'

const ROW_CAP = 120 // rendered rows; the 24h window itself can hold thousands

const BANDS = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'threat', label: `≥ ${THREAT_THRESHOLD * 100}%`, test: (p) => p >= THREAT_THRESHOLD },
  { id: 'high', label: '≥ 80%', test: (p) => p >= 0.8 },
]

const SORTS = { latest: 'Time', risk: 'Score' }

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

  // Sortable column heading; the arrow only shows on the active sort.
  const sortHead = (id, className = '') => (
    <button
      onClick={() => setSort(id)}
      aria-sort={sort === id ? 'descending' : 'none'}
      className={`th inline-flex cursor-pointer items-center gap-1 bg-transparent p-0 hover:text-ink ${sort === id ? 'text-ink' : ''} ${className}`}
    >
      {SORTS[id]}
      {sort === id && <ArrowDown size={12} aria-hidden="true" />}
    </button>
  )

  return (
    <section className="sheet min-w-0" aria-label="Scored transfers">
      <div className="sheet-head flex-wrap gap-y-2 py-2">
        <h2 className="sheet-title">Results</h2>
        <span className="text-[12.5px] text-ink-3">
          {rows.length > ROW_CAP ? `first ${ROW_CAP} of ${rows.length.toLocaleString()}` : rows.length.toLocaleString()}
          {rows.length !== transactions.length && ` · ${transactions.length.toLocaleString()} total`}
        </span>
        <div className="ml-auto flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <fieldset className="seg m-0 p-0">
            <legend className="sr-only">Score band</legend>
            {BANDS.map((b) => (
              <label key={b.id} className="seg-opt">
                <input type="radio" name="rc-band" checked={band === b.id} onChange={() => setBand(b.id)} />
                {b.label}
              </label>
            ))}
          </fieldset>
          <div className="relative order-first min-w-0 basis-full sm:order-none sm:basis-auto">
            <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <input
              className="input mono min-h-[32px] pl-8 text-[13px] sm:w-[230px]"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by hash or address"
              aria-label="Filter by hash or address"
            />
          </div>
          <button
            className="btn btn-secondary min-h-[32px] px-3 text-[12.5px] md:hidden"
            onClick={() => setSort((s) => (s === 'latest' ? 'risk' : 'latest'))}
          >
            Sort: {sort === 'latest' ? 'newest' : 'highest score'}
          </button>
        </div>
      </div>

      <div className="feed-grid hidden border-b border-line px-4 py-2 md:grid">
        {sortHead('latest')}
        <div className="th">Transaction</div>
        <div className="th">From → to</div>
        <div className="th text-right">Value</div>
        <div className="th">Range · ref &lt; {THREAT_THRESHOLD * 100}%</div>
        {sortHead('risk', 'justify-self-end')}
        <div className="th">Flag</div>
      </div>

      <div>
        {rows.slice(0, ROW_CAP).map((t) => {
          const p = t.probability || 0
          const flagged = p >= THREAT_THRESHOLD
          return (
            <button
              key={t.tx_hash}
              onClick={() => onSelect(t.tx_hash)}
              aria-current={selected === t.tx_hash ? 'true' : undefined}
              className="row-btn feed-grid items-center px-4 py-2"
            >
              <span className="text-[12.5px] text-ink-3" style={{ gridArea: 'age' }}>{relativeTime(t.timestamp)}</span>
              <span className="min-w-0" style={{ gridArea: 'tx' }}>
                <span className="mono block truncate text-[13px] font-semibold">{shortHash(t.tx_hash, 16)}</span>
                <span className="mono block truncate text-[12.5px] text-ink-2 md:hidden"><span className="font-sans">from </span>{shortAddr(t.address)}</span>
              </span>
              <span className="mono min-w-0 truncate text-[12.5px] text-ink-2" style={{ gridArea: 'to' }}>
                <span className="font-sans md:hidden">to </span><span className="hidden md:inline">{shortAddr(t.address)} → </span>{t.to_address ? shortAddr(t.to_address) : 'contract creation'}
              </span>
              <span className="mono text-right text-[13px]" style={{ gridArea: 'value' }}>{formatEth(t.value)} Ξ</span>
              <RangeBar value={p} history={seriesFor(index, t.address)} className="min-w-0" style={{ gridArea: 'bar' }} />
              <span
                className={`mono text-right text-[13.5px] ${flagged ? 'font-bold' : ''}`}
                style={{ gridArea: 'pct', color: riskColor(p) }}
              >
                {formatPct(p)}
              </span>
              <span data-flag className="text-[12.5px] font-bold" style={{ gridArea: 'flag', color: riskColor(p) }}>{riskFlag(p)}</span>
            </button>
          )
        })}

        {rows.length === 0 && (
          <EmptyState
            title={transactions.length ? 'Nothing matches that filter' : 'Waiting for the first scored block'}
            description={transactions.length ? 'Try a different hash, address, or score band.' : null}
          />
        )}
      </div>
    </section>
  )
}
