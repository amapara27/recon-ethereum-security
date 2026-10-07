import { ChevronDown } from 'lucide-react'
import { severityColor } from '../lib/risk'

// One finding from the contract-analyzer report. The API returns
// { type, severity, description, line_number } — no snippet or remediation field.
export default function FindingRow({ finding, open, onToggle }) {
  const color = severityColor(finding.severity)
  const loc = finding.line_number && finding.line_number !== 'N/A' ? `line ${finding.line_number}` : ''

  return (
    <div className="border-b border-line">
      <button
        onClick={onToggle}
        aria-expanded={open}
        className="grid w-full cursor-pointer grid-cols-[76px_minmax(0,1fr)_auto_16px] items-center gap-3 bg-transparent px-5 py-3 text-left hover:bg-sheet-2"
      >
        <span className="text-[12.5px] font-bold" style={{ color }}>{finding.severity || 'Info'}</span>
        <span className="truncate text-[14px] font-semibold">{finding.type || 'Finding'}</span>
        <span className="mono text-[12.5px] text-ink-3">{loc}</span>
        <ChevronDown size={15} className={`text-ink-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <p className="max-w-[72ch] pb-4 pl-[108px] pr-5 text-[13.5px] leading-[1.6] text-ink-2 text-pretty max-sm:pl-5">
          {finding.description}
        </p>
      )}
    </div>
  )
}
