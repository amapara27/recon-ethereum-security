import { riskColor, THREAT_THRESHOLD } from '../lib/risk'
import { formatPct } from '../lib/format'

const pos = (p) => `${Math.max(0, Math.min(1, p || 0)) * 100}%`
const HISTORY_CAP = 16 // dots beyond this stop being readable at row size
const MIN_GAP = 0.045 // ≈ a dot diameter + 2px at row width; closer dots merge, none sit under the marker

function spread(history, value) {
  const out = []
  for (const p of [...history].sort((a, b) => a - b)) {
    if (Math.abs(p - (value || 0)) < MIN_GAP) continue
    if (out.length && p - out[out.length - 1] < MIN_GAP) continue
    out.push(p)
  }
  return out
}

// A fraud probability on the fixed 0–100% scale every score in the app shares.
// `history` = this address's earlier scores in the window, drawn as hollow dots.
// `axis` adds the scale labels underneath (used where the bar is large).
export default function RangeBar({ value, history = [], height = 12, axis = false, className = '', style }) {
  const color = riskColor(value)
  const dots = spread(history.slice(-HISTORY_CAP), value)

  return (
    <div className={className} style={style}>
      <div
        className="rb"
        style={{ height }}
        role="img"
        aria-label={`Score ${formatPct(value)} on a 0–100% scale, threshold ${THREAT_THRESHOLD * 100}%`}
      >
        <span className="rb-ref" style={{ width: pos(THREAT_THRESHOLD) }} />
        <span className="rb-tick" style={{ left: pos(THREAT_THRESHOLD) }} />
        <span className="rb-tick dashed" style={{ left: '80%' }} />
        {dots.map((p, i) => <span key={i} className="rb-dot" style={{ left: pos(p) }} />)}
        <span className="rb-mark" style={{ left: pos(value), background: color }} />
      </div>
      {axis && (
        <div className="mono relative mt-1.5 h-4 text-[11px] text-ink-3" aria-hidden="true">
          <span className="absolute left-0">0</span>
          <span className="absolute -translate-x-1/2" style={{ left: '50%' }}>50</span>
          <span className="absolute -translate-x-1/2" style={{ left: '80%' }}>80</span>
          <span className="absolute right-0">100%</span>
        </div>
      )}
    </div>
  )
}
