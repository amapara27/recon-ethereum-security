import { getScoreBand } from '../lib/risk'

// SVG ring for the contract auditor's 0..100 safety score.
export default function SafetyScoreRing({ score = 0, size = 132 }) {
  const band = getScoreBand(score)
  const r = size / 2 - 8
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(100, score))
  const offset = c - (pct / 100) * c

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth="8" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={band.stroke}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 600ms ease' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={`font-mono text-3xl font-semibold tabular ${band.text}`}>{Math.round(pct)}</span>
        <span className={`text-xs font-medium uppercase tracking-wide ${band.text}`}>{band.label}</span>
      </div>
    </div>
  )
}
