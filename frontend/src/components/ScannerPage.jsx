import { useMemo } from 'react'
import { Activity, ShieldAlert, Gauge, TrendingUp } from 'lucide-react'
import StatTile from './StatTile'
import LiveScanner from './LiveScanner'
import ThreatsPanel from './ThreatsPanel'
import { THREAT_THRESHOLD } from '../lib/risk'
import { formatPct } from '../lib/format'

export default function ScannerPage({ alerts }) {
  // KPIs derived entirely from the live alert feed — no fabricated numbers.
  const kpis = useMemo(() => {
    const n = alerts.length
    const probs = alerts.map((a) => a.probability || 0)
    const threats = probs.filter((p) => p >= THREAT_THRESHOLD).length
    const avg = n ? probs.reduce((s, p) => s + p, 0) / n : 0
    const peak = n ? Math.max(...probs) : 0
    return { n, threats, avg, peak }
  }, [alerts])

  return (
    <div className="flex flex-col gap-4 lg:gap-6">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile label="Scanned (24h)" value={kpis.n} sublabel="addresses scored" icon={Activity} tone="accent" />
        <StatTile label="Active threats" value={kpis.threats} sublabel={`≥ ${THREAT_THRESHOLD * 100}% risk`} icon={ShieldAlert} tone={kpis.threats ? 'high' : 'safe'} />
        <StatTile label="Avg risk" value={formatPct(kpis.avg)} sublabel="across window" icon={Gauge} />
        <StatTile label="Peak risk" value={formatPct(kpis.peak)} sublabel="highest scored" icon={TrendingUp} tone={kpis.peak >= 0.8 ? 'high' : 'med'} />
      </div>

      <div className="grid min-h-0 gap-4 lg:grid-cols-[2fr_1fr] lg:gap-6">
        <div className="min-h-[420px] lg:h-[calc(100dvh-15rem)]">
          <LiveScanner transactions={alerts} />
        </div>
        <div className="min-h-[420px] lg:h-[calc(100dvh-15rem)]">
          <ThreatsPanel transactions={alerts} />
        </div>
      </div>
    </div>
  )
}
