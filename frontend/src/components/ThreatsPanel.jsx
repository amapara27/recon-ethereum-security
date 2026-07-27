import { ShieldAlert, ShieldCheck } from 'lucide-react'
import EmptyState from './EmptyState'
import { shortAddr, formatEth, relativeTime } from '../lib/format'
import { riskColor, THREAT_THRESHOLD } from '../lib/risk'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`

export default function ThreatsPanel({ threats, onSelect, compact }) {
  return (
    <div className="card elev-sm overflow-hidden rounded-md p-0">
      <div className="flex items-center gap-2 border-b border-line px-3.5 py-[11px]">
        <ShieldAlert size={15} style={{ color: 'var(--risk-high)' }} />
        <span className="text-sm" style={{ fontFamily: 'var(--font-heading)' }}>Active threats</span>
        <span className="mono ml-auto text-[11px]" style={{ color: MUTED(45) }}>≥{THREAT_THRESHOLD * 100}%</span>
      </div>

      <div className="rc-scroll overflow-auto" style={{ maxHeight: compact ? '240px' : 'max(320px, calc(100dvh - 330px))' }}>
        {threats.slice(0, 40).map((t) => {
          const color = riskColor(t.probability)
          return (
            <div
              key={t.tx_hash}
              onClick={() => onSelect(t.tx_hash)}
              className="rc-row cursor-pointer border-b px-3.5 py-2.5"
              style={{ boxShadow: `inset 3px 0 0 ${color}`, borderColor: MUTED(6) }}
            >
              <div className="flex items-center gap-2">
                <span className="text-[10.5px] uppercase tracking-[0.07em]" style={{ color }}>
                  {(t.probability || 0) >= 0.8 ? 'High-risk transfer' : 'Elevated risk'}
                </span>
                <span className="mono ml-auto text-[11.5px]" style={{ color }}>
                  {((t.probability || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mono mt-[5px] truncate text-[11.5px]" style={{ color: MUTED(70) }}>
                {shortAddr(t.address)} → {shortAddr(t.to_address)}
              </div>
              <div className="mono mt-1 text-[11px]" style={{ color: MUTED(40) }}>
                {relativeTime(t.timestamp)} · {formatEth(t.value)} Ξ
              </div>
            </div>
          )
        })}

        {threats.length === 0 && (
          <EmptyState
            icon={ShieldCheck}
            iconColor="var(--risk-safe)"
            title="Nothing above threshold"
            className="px-5 py-10"
          />
        )}
      </div>
    </div>
  )
}
