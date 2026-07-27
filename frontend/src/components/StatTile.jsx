import Sparkline from './Sparkline'

// KPI tile with a trend line. Both the value and the series come from the live alert feed.
export default function StatTile({ label, value, sublabel, icon: Icon, color = 'var(--color-text)', series }) {
  return (
    <div className="card elev-sm rounded-md px-[15px] py-[13px]">
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-[0.07em] text-muted">{label}</span>
        {Icon && <Icon size={14} className="text-muted" aria-hidden="true" />}
      </div>
      <div className="mt-2 flex items-end justify-between gap-2.5">
        <div className="mono text-[25px] tracking-[-0.02em]" style={{ color }}>{value}</div>
        <Sparkline values={series} w={72} h={24} color={color} area strokeWidth={1.5} opacity={1} />
      </div>
      {sublabel && <div className="mt-1 text-[11.5px] text-muted">{sublabel}</div>}
    </div>
  )
}
