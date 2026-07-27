import { getScoreBand } from '../lib/risk'

// SVG ring for the contract auditor's 0..100 safety score.
export default function SafetyScoreRing({ score = 0, size = 104 }) {
  const band = getScoreBand(score)
  const r = size / 2 - 6
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, score))

  return (
    <div className="relative flex-none" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="color-mix(in srgb, currentColor 12%, transparent)" strokeWidth="7" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={band.color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c - (pct / 100) * c}
          style={{ transition: 'stroke-dashoffset 600ms ease' }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="mono text-[26px]" style={{ color: band.color }}>{Math.round(pct)}</div>
          <div className="text-[10px] uppercase tracking-[0.09em]" style={{ color: band.color }}>{band.label}</div>
        </div>
      </div>
    </div>
  )
}
