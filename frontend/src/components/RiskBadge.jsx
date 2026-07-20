import { getRisk } from '../lib/risk'
import { formatPct } from '../lib/format'

// Risk is never encoded by color alone — the % and a label always accompany the color.
export default function RiskBadge({ probability, showPct = true, size = 'sm' }) {
  const risk = getRisk(probability)
  const pad = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${risk.text} ${risk.border} ${pad}`}
      title={`Fraud probability ${formatPct(probability)}`}
    >
      <span className={`size-1.5 rounded-full ${risk.bg}`} aria-hidden="true" />
      {showPct ? <span className="tabular font-mono">{formatPct(probability)}</span> : risk.label}
    </span>
  )
}
