import EmptyState from './EmptyState'
import { shortAddr, formatEth, formatPct, relativeTime } from '../lib/format'
import { riskColor, riskFlag, THREAT_THRESHOLD } from '../lib/risk'

// Every result at or above the threshold, highest first.
export default function ThreatsPanel({ threats, selected, onSelect, compact }) {
  return (
    <section className="sheet" aria-label="Flagged results">
      <div className="sheet-head">
        <h2 className="sheet-title">Flagged</h2>
        <span className="mono text-[12.5px] text-ink-3">{threats.length.toLocaleString()}</span>
        <span className="ml-auto text-[12.5px] text-ink-3">≥ {THREAT_THRESHOLD * 100}%, highest first</span>
      </div>

      <div className="scroll overflow-auto" style={{ maxHeight: compact ? 260 : 'max(360px, calc(100dvh - 190px))' }}>
        {threats.slice(0, 40).map((t) => {
          const color = riskColor(t.probability)
          return (
            <button
              key={t.tx_hash}
              onClick={() => onSelect(t.tx_hash)}
              aria-current={selected === t.tx_hash ? 'true' : undefined}
              className="row-btn grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-0.5 px-4 py-2.5"
            >
              <span className="text-[13px] font-bold" style={{ color }}>{riskFlag(t.probability)}</span>
              <span className="mono text-right text-[13.5px] font-bold" style={{ color }}>{formatPct(t.probability)}</span>
              <span className="mono truncate text-[12.5px] text-ink-2">
                {shortAddr(t.address)} → {t.to_address ? shortAddr(t.to_address) : 'contract creation'}
              </span>
              <span className="text-right text-[12.5px] text-ink-3">{relativeTime(t.timestamp)}</span>
              <span className="mono col-span-2 text-[12.5px] text-ink-3">{formatEth(t.value)} Ξ</span>
            </button>
          )
        })}

        {threats.length === 0 && (
          <EmptyState title="Nothing above the threshold" />
        )}
      </div>
    </section>
  )
}
