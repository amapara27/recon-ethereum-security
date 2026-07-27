import { ChevronDown } from 'lucide-react'
import { severityColor } from '../lib/risk'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`

// One finding from the contract-analyzer report. The API returns
// { type, severity, description, line_number } — no snippet or remediation field.
export default function FindingRow({ finding, open, onToggle }) {
  const color = severityColor(finding.severity)
  const loc = finding.line_number && finding.line_number !== 'N/A' ? `L${finding.line_number}` : ''

  return (
    <div className="border-b" style={{ boxShadow: `inset 3px 0 0 ${color}`, borderColor: MUTED(6) }}>
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center gap-2.5 border-0 bg-transparent px-[15px] py-[11px] text-left"
        style={{ color: 'inherit', font: 'inherit' }}
      >
        <span
          className="mono rounded-sm border px-[7px] py-0.5 text-[10px] uppercase tracking-[0.06em]"
          style={{ borderColor: color, color }}
        >
          {finding.severity || 'info'}
        </span>
        <span className="min-w-0 flex-1 truncate text-[13.5px]">{finding.type || 'Finding'}</span>
        {loc && <span className="mono text-[11px]" style={{ color: MUTED(45) }}>{loc}</span>}
        <ChevronDown size={13} className={`shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} style={{ color: MUTED(45) }} />
      </button>
      {open && (
        <p className="m-0 px-[15px] pb-[13px] text-[13px] leading-[1.6] text-pretty" style={{ color: MUTED(72) }}>
          {finding.description}
        </p>
      )}
    </div>
  )
}
