// Compact KPI tile. Values are always derived from real alert data (never fabricated).
export default function StatTile({ label, value, sublabel, icon: Icon, tone = 'neutral' }) {
  const toneText = {
    neutral: 'text-ink',
    accent: 'text-accent',
    high: 'text-risk-high',
    med: 'text-risk-med',
    safe: 'text-risk-safe',
  }[tone]

  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow)]">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        {Icon && <Icon size={16} className="text-muted" aria-hidden="true" />}
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold tabular ${toneText}`}>{value}</div>
      {sublabel && <div className="mt-0.5 text-xs text-muted">{sublabel}</div>}
    </div>
  )
}
