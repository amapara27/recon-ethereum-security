import { riskColor } from '../lib/risk'
import { formatPct } from '../lib/format'

// Risk is never encoded by color alone — the % always accompanies the color.
export default function RiskBadge({ probability }) {
  const color = riskColor(probability)
  return (
    <span
      className="mono inline-flex items-center gap-[5px] rounded-full border px-2 py-[2px] text-[11.5px]"
      style={{ borderColor: color, color }}
      title={`Fraud probability ${formatPct(probability)}`}
    >
      <span className="size-[5px] rounded-full" style={{ background: color }} aria-hidden="true" />
      {formatPct(probability)}
    </span>
  )
}
