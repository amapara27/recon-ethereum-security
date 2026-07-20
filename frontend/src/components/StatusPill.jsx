// Live backend connection indicator, driven by the polling hook.
export default function StatusPill({ status }) {
  const map = {
    online: { dot: 'bg-risk-safe', label: 'Live', ring: 'shadow-[0_0_0_3px_rgba(34,197,94,0.15)]' },
    offline: { dot: 'bg-risk-high', label: 'Offline', ring: '' },
    connecting: { dot: 'bg-risk-med', label: 'Connecting', ring: '' },
  }
  const s = map[status] || map.connecting
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-xs font-medium text-ink">
      <span className={`size-2 rounded-full ${s.dot} ${s.ring} ${status === 'online' ? 'animate-pulse' : ''}`} aria-hidden="true" />
      {s.label}
    </span>
  )
}
