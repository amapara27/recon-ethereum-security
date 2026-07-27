import { useMemo } from 'react'
import { Activity, ShieldAlert, Gauge, TrendingUp } from 'lucide-react'
import StatTile from './StatTile'
import LiveScanner from './LiveScanner'
import ThreatsPanel from './ThreatsPanel'
import AddressPanel from './AddressPanel'
import { THREAT_THRESHOLD, riskColor } from '../lib/risk'
import { formatPct } from '../lib/format'
import { timeBuckets, meanProb, peakProb } from '../lib/series'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`
const WINDOW_MS = 24 * 60 * 60 * 1000 // the API window: alerts from the last 24 hours
const TREND_BUCKETS = 14
const STRIP_BUCKETS = 56

export default function ScannerPage({ alerts, selectedHash, onSelect, onAudit, watched, onWatch }) {
  const { kpis, strip, threats } = useMemo(() => {
    const probs = alerts.map((a) => a.probability || 0)
    const flagged = alerts.filter((a) => (a.probability || 0) >= THREAT_THRESHOLD)
    const peak = probs.length ? Math.max(...probs) : 0
    const mean = probs.length ? probs.reduce((s, p) => s + p, 0) / probs.length : 0
    const trend = (fn) => timeBuckets(alerts, TREND_BUCKETS, WINDOW_MS, fn)

    return {
      threats: [...flagged].sort((a, b) => (b.probability || 0) - (a.probability || 0)),
      kpis: [
        { label: 'Scanned', value: alerts.length.toLocaleString(), sublabel: 'transfers scored, 24h', icon: Activity, color: 'var(--color-text)', series: trend((r) => r.length) },
        { label: 'Active threats', value: flagged.length.toLocaleString(), sublabel: `≥ ${THREAT_THRESHOLD * 100}% fraud probability`, icon: ShieldAlert, color: flagged.length ? 'var(--risk-high)' : 'var(--risk-safe)', series: trend((r) => r.filter((a) => (a.probability || 0) >= THREAT_THRESHOLD).length) },
        { label: 'Mean risk', value: formatPct(mean), sublabel: 'across the window', icon: Gauge, color: 'var(--color-accent)', series: trend(meanProb) },
        { label: 'Peak risk', value: formatPct(peak), sublabel: 'highest single score', icon: TrendingUp, color: riskColor(peak), series: trend(peakProb) },
      ],
      strip: timeBuckets(alerts, STRIP_BUCKETS, WINDOW_MS, (rows) => ({ v: meanProb(rows), n: rows.length })),
    }
  }, [alerts])

  const selected = selectedHash ? alerts.find((a) => a.tx_hash === selectedHash) : null

  return (
    <div className="flex flex-col gap-3.5">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        {kpis.map((k) => <StatTile key={k.label} {...k} />)}
      </div>

      {/* Risk over time — the mock binned by block; the feed gives timestamps, so we bin by time */}
      <div className="card elev-sm rounded-md px-[15px] py-[13px]">
        <div className="flex flex-wrap items-center gap-2.5">
          <span className="text-[11px] uppercase tracking-[0.07em]" style={{ color: MUTED(55) }}>Risk over time</span>
          <span className="mono text-[11px]" style={{ color: MUTED(40) }}>last 24h · {STRIP_BUCKETS} buckets</span>
          <div className="ml-auto flex items-center gap-1.5 text-[11px]" style={{ color: MUTED(45) }}>
            low
            <span
              className="h-1.5 w-[54px] rounded-[3px]"
              style={{ background: 'linear-gradient(to right, var(--risk-safe), var(--risk-med), var(--risk-high))' }}
            />
            high
          </div>
        </div>
        <div className="mt-[11px] flex items-end gap-0.5">
          {strip.map((b, i) => (
            <div
              key={i}
              title={b.n ? `${b.n} transfers · mean risk ${formatPct(b.v)}` : 'no transfers in this bucket'}
              className="flex-1 rounded-[2px]"
              style={{
                height: `${(8 + b.v * 30).toFixed(0)}px`,
                minHeight: '4px',
                background: b.n ? riskColor(b.v) : MUTED(15),
                opacity: b.n ? (0.35 + b.v * 0.65).toFixed(2) : 1,
              }}
            />
          ))}
        </div>
      </div>

      <div className="grid items-start gap-3.5 xl:grid-cols-[minmax(0,1fr)_344px]">
        <LiveScanner transactions={alerts} selected={selectedHash} onSelect={onSelect} />

        <div className="flex flex-col gap-3.5 xl:sticky xl:top-0">
          {selected && (
            <AddressPanel
              tx={selected}
              alerts={alerts}
              onClear={() => onSelect(null)}
              onAudit={onAudit}
              watched={watched.includes(selected.address?.toLowerCase())}
              onWatch={onWatch}
            />
          )}
          <ThreatsPanel threats={threats} onSelect={onSelect} compact={!!selected} />
        </div>
      </div>
    </div>
  )
}
