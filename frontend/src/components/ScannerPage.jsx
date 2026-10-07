import { useMemo } from 'react'
import LiveScanner from './LiveScanner'
import ThreatsPanel from './ThreatsPanel'
import AddressPanel from './AddressPanel'
import { THREAT_THRESHOLD, riskColor } from '../lib/risk'
import { formatPct } from '../lib/format'
import { timeBuckets } from '../lib/series'

const WINDOW_MS = 24 * 60 * 60 * 1000 // the API window: alerts from the last 24 hours
const HOURS = 24

export default function ScannerPage({ alerts, selectedHash, onSelect, onAudit, watched, onWatch }) {
  const { summary, hours, threats } = useMemo(() => {
    const probs = alerts.map((a) => a.probability || 0)
    const flagged = alerts.filter((a) => (a.probability || 0) >= THREAT_THRESHOLD)
    const high = flagged.filter((a) => a.probability >= 0.8).length
    const peak = probs.length ? Math.max(...probs) : 0
    const mean = probs.length ? probs.reduce((s, p) => s + p, 0) / probs.length : 0
    const share = alerts.length ? `, ${((flagged.length / alerts.length) * 100).toFixed(1)}% of scored` : ''

    return {
      threats: [...flagged].sort((a, b) => (b.probability || 0) - (a.probability || 0)),
      summary: [
        // Counts stay ink; only a result (the peak) earns flag colour.
        { label: 'Scored', value: alerts.length.toLocaleString(), note: 'in 24h' },
        { label: 'Flagged', value: flagged.length.toLocaleString(), note: `≥ ${THREAT_THRESHOLD * 100}%${share}` },
        { label: 'High', value: high.toLocaleString(), note: '≥ 80%' },
        { label: 'Mean', value: formatPct(mean) },
        { label: 'Peak', value: formatPct(peak), color: peak >= THREAT_THRESHOLD ? riskColor(peak) : null },
      ],
      hours: timeBuckets(alerts, HOURS, WINDOW_MS, (rows) => {
        const p = rows.map((a) => a.probability || 0)
        return { n: rows.length, high: p.filter((x) => x >= 0.8).length, med: p.filter((x) => x >= THREAT_THRESHOLD && x < 0.8).length }
      }),
    }
  }, [alerts])

  const selected = selectedHash ? alerts.find((a) => a.tx_hash === selectedHash) : null
  const maxHour = Math.max(1, ...hours.map((h) => h.n))

  return (
    <div className="flex flex-col gap-4">
      <section className="sheet" aria-label="Window summary">
        {/* One ruled specimen line at table size, ranked by weight */}
        <dl className="m-0 flex flex-wrap gap-x-7 gap-y-1.5 border-b border-line-strong px-4 py-3 text-[14px]">
          {summary.map((s) => (
            <div key={s.label} className="flex items-baseline gap-2">
              <dt className="text-ink-2">{s.label}</dt>
              <dd className="mono m-0 font-semibold" style={{ color: s.color || undefined }}>{s.value}</dd>
              {s.note && <dd className="m-0 text-[12.5px] text-ink-3">{s.note}</dd>}
            </div>
          ))}
        </dl>

        {/* Count per hour on a zero-based scale; the flagged share is stacked in its flag colour */}
        <figure className="m-0 px-4 py-3">
          <figcaption className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="th">Scored per hour</span>
            <span className="flex items-center gap-1.5 text-[12px] text-ink-3">
              <span className="inline-block size-2" style={{ background: 'var(--risk-med)' }} />elevated
              <span className="ml-1.5 inline-block size-2" style={{ background: 'var(--risk-high)' }} />high
            </span>
          </figcaption>
          <div className="mt-2 flex h-[60px] items-end gap-[3px]">
            {hours.map((h, i) => (
              <div
                key={i}
                className="flex flex-1 flex-col-reverse"
                style={{ height: `${(h.n / maxHour) * 100}%`, minHeight: 1, background: h.n ? 'color-mix(in srgb, var(--ink) 22%, transparent)' : 'var(--rule)' }}
                title={`${h.n} scored · ${h.med + h.high} flagged`}
              >
                {h.high > 0 && <span style={{ height: `${(h.high / h.n) * 100}%`, background: 'var(--risk-high)' }} />}
                {h.med > 0 && <span style={{ height: `${(h.med / h.n) * 100}%`, background: 'var(--risk-med)' }} />}
              </div>
            ))}
          </div>
          <div className="mt-1 flex justify-between text-[12px] text-ink-3" aria-hidden="true">
            <span>24h ago</span><span>12h</span><span>now</span>
          </div>
        </figure>
      </section>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <LiveScanner transactions={alerts} selected={selectedHash} onSelect={onSelect} />

        <div className="flex flex-col gap-4 xl:sticky xl:top-0">
          {selected && (
            <AddressPanel
              key={selected.tx_hash}
              tx={selected}
              alerts={alerts}
              onClear={() => onSelect(null)}
              onAudit={onAudit}
              watched={watched.includes(selected.address?.toLowerCase())}
              onWatch={onWatch}
            />
          )}
          <ThreatsPanel threats={threats} selected={selectedHash} onSelect={onSelect} compact={!!selected} />
        </div>
      </div>
    </div>
  )
}
