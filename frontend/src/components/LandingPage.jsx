import { useMemo } from 'react'
import { ShieldCheck, ArrowRight, ArrowUpRight, FileSearch, LockOpen, Database, Radar, Eye } from 'lucide-react'
import ThemeToggle from './ThemeToggle'
import Sparkline from './Sparkline'
import { shortHash, shortAddr, formatPct, toMs } from '../lib/format'
import { riskColor, THREAT_THRESHOLD } from '../lib/risk'
import { indexByAddress, seriesFor } from '../lib/series'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`

// Model numbers are the ones the README documents for the trained classifier.
const STATS = [
  { value: '0.99', label: 'ROC-AUC, held-out labels' },
  { value: '814', label: 'Behavioural features per address' },
  { value: '0.96', label: 'Recall on held-out fraud' },
  { value: '24/7', label: 'Mainnet coverage, every block' },
]

const STEPS = [
  ['01', 'Ingest', 'Every transaction in a new block, straight off the node.'],
  ['02', 'Featurise', '814 behavioural features per address, recomputed on each touch.'],
  ['03', 'Score & surface', 'Fraud probability written to the alert feed the moment it lands.'],
]

const PILLARS = [
  { icon: Radar, title: 'Live scanner', body: 'Dense feed of scored transfers with severity bands, a rolling risk strip and one-click drill-down into any address.' },
  { icon: FileSearch, title: 'Contract auditor', body: 'Pulls verified source from Etherscan and returns a graded report: findings by severity, affected lines, executive summary.' },
  { icon: Eye, title: 'Watchlist', body: 'Pin addresses and watch their score drift across the window as the feed re-scores them.' },
]

export default function LandingPage({ onEnter, onAudit, alerts, theme, onToggleTheme }) {
  const { preview, perMinute, bands, index } = useMemo(() => {
    // Rate over the minute ending at the newest alert — derived from the feed rather than
    // the wall clock, so it stays stable across re-renders.
    const newest = alerts.length ? toMs(alerts[0].timestamp) : 0
    const counts = { safe: 0, med: 0, high: 0 }
    for (const a of alerts) {
      const p = a.probability || 0
      if (p >= 0.8) counts.high++
      else if (p >= THREAT_THRESHOLD) counts.med++
      else counts.safe++
    }
    const max = Math.max(counts.safe, counts.med, counts.high, 1)
    return {
      preview: alerts.slice(0, 6),
      index: indexByAddress(alerts),
      perMinute: alerts.filter((a) => toMs(a.timestamp) >= newest - 60_000).length,
      bands: [
        { label: 'Low risk', hint: `< ${THREAT_THRESHOLD * 100}%`, n: counts.safe, color: 'var(--risk-safe)', w: `${(counts.safe / max) * 100}%` },
        { label: 'Elevated', hint: `≥ ${THREAT_THRESHOLD * 100}%`, n: counts.med, color: 'var(--risk-med)', w: `${(counts.med / max) * 100}%` },
        { label: 'High risk', hint: '≥ 80%', n: counts.high, color: 'var(--risk-high)', w: `${(counts.high / max) * 100}%` },
      ],
    }
  }, [alerts])

  return (
    <div className="mx-auto max-w-[1240px] px-5 pb-24 sm:px-8">
      <header className="flex flex-wrap items-center gap-x-7 gap-y-3 pt-[22px]">
        <div className="mr-auto flex items-center gap-[9px]">
          <span className="grid size-[30px] place-items-center rounded-lg border border-accent text-accent">
            <ShieldCheck size={17} />
          </span>
          <span className="text-[17px] tracking-[-0.01em]" style={{ fontFamily: 'var(--font-heading)' }}>Recon</span>
          <span className="tag tag-neutral ml-1">beta</span>
        </div>
        <nav className="hidden gap-[22px] text-[13px] md:flex" style={{ color: MUTED(62) }}>
          <a href="#model" className="no-underline" style={{ color: 'inherit' }}>Model</a>
          <a href="#pillars" className="no-underline" style={{ color: 'inherit' }}>Scanner</a>
          <a href="#pillars" className="no-underline" style={{ color: 'inherit' }}>Auditor</a>
        </nav>
        <div className="flex gap-2">
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
          <button className="btn btn-primary" onClick={onEnter}>Launch app<ArrowRight size={14} /></button>
        </div>
      </header>

      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="grid items-center gap-14 py-16 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:pb-[72px] lg:pt-[88px]">
        <div>
          <div className="inline-flex items-center gap-[7px] text-[11px] uppercase tracking-[0.09em]" style={{ color: MUTED(60) }}>
            <span className="rc-pulse size-[5px] rounded-full" style={{ background: 'var(--risk-safe)' }} aria-hidden="true" />
            Scoring Ethereum mainnet · {alerts.length.toLocaleString()} transfers in the last 24h
          </div>
          <h1 className="mono mt-5 max-w-[15ch] text-[40px] leading-[1.06] tracking-[-0.045em] sm:text-[58px]" style={{ fontWeight: 500 }}>
            Fraud, scored at block time.
          </h1>
          <p className="mt-[22px] max-w-[46ch] text-base leading-[1.6] text-pretty" style={{ color: MUTED(72) }}>
            Recon scores every address touching mainnet with a random-forest classifier over 814 behavioural
            features, and audits verified Solidity for reentrancy, honeypots and owner escape hatches.
          </p>
          <div className="mt-[30px] flex flex-wrap gap-2.5">
            <button className="btn btn-primary px-[18px] py-2.5" onClick={onEnter}>
              Open live scanner<ArrowRight size={14} />
            </button>
            <button className="btn btn-secondary px-[18px] py-2.5" onClick={onAudit}>
              <FileSearch size={15} />Audit a contract
            </button>
          </div>
          <div className="mt-[34px] flex flex-wrap gap-5 text-xs" style={{ color: MUTED(55) }}>
            <span className="flex items-center gap-1.5"><LockOpen size={13} />Read-only · no wallet connection</span>
            <span className="flex items-center gap-1.5"><Database size={13} />Etherscan-verified sources</span>
          </div>
        </div>

        {/* Live feed preview — the same rows the scanner shows, straight from the API */}
        <div className="card elev-md rc-rise overflow-hidden rounded-lg p-0">
          <div className="flex items-center gap-2 border-b border-line px-3.5 py-[11px]">
            <span className="rc-pulse size-1.5 rounded-full" style={{ background: 'var(--risk-safe)' }} aria-hidden="true" />
            <span className="text-xs uppercase tracking-[0.06em]" style={{ color: MUTED(60) }}>Live feed</span>
            <span className="mono ml-auto text-[11px]" style={{ color: MUTED(45) }}>{perMinute} scored / min</span>
          </div>
          <div>
            {preview.length === 0 && (
              <div className="px-3.5 py-10 text-center text-[13px]" style={{ color: MUTED(55) }}>
                Waiting for the first scored block…
              </div>
            )}
            {preview.map((t) => {
              const color = riskColor(t.probability)
              return (
                <div
                  key={t.tx_hash}
                  className="grid grid-cols-[1fr_auto] items-center gap-3 border-b px-3.5 py-[11px]"
                  style={{ boxShadow: `inset 3px 0 0 ${color}`, borderColor: MUTED(7) }}
                >
                  <div className="min-w-0">
                    <div className="mono truncate text-xs text-accent">{shortHash(t.tx_hash, 14)}</div>
                    <div className="mono mt-0.5 text-[11px]" style={{ color: MUTED(50) }}>
                      {shortAddr(t.address)} → {shortAddr(t.to_address)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2.5">
                    <Sparkline
                      values={seriesFor(index, t.address)}
                      w={46}
                      h={16}
                      color={color}
                      opacity={0.8}
                      strokeWidth={1.4}
                    />
                    <span className="mono min-w-[46px] text-right text-xs" style={{ color }}>{formatPct(t.probability)}</span>
                  </div>
                </div>
              )
            })}
          </div>
          <button className="btn btn-ghost w-full rounded-none py-[11px] text-xs" onClick={onEnter}>
            Open full scanner<ArrowUpRight size={13} />
          </button>
        </div>
      </section>

      {/* ── Model stats band ─────────────────────────────────────────── */}
      <section
        className="-mx-5 rounded-lg px-5 py-11 sm:-mx-8 sm:px-8"
        style={{ background: 'var(--color-section)', color: '#f3f5fe' }}
      >
        <div className="mx-auto grid max-w-[1176px] grid-cols-2 gap-8 lg:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label}>
              <div className="mono text-[38px] tracking-[-0.03em]">{s.value}</div>
              <div className="mt-1.5 text-xs" style={{ color: 'color-mix(in srgb, #f3f5fe 70%, transparent)' }}>{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How the model works ──────────────────────────────────────── */}
      <section id="model" className="grid items-start gap-16 pt-[88px] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <div>
          <h6 className="text-accent">How the model works</h6>
          <h2 className="mt-3.5 tracking-[-0.02em]">Behaviour, not blocklists.</h2>
          <p className="mt-[18px] text-[15px] leading-[1.65] text-pretty" style={{ color: MUTED(70) }}>
            Blocklists only catch addresses that already burned someone. Recon fingerprints how an address
            transacts — timing, counterparty churn, value patterns, contract-call shape — so a fresh wallet
            draining a victim scores high on its first hostile transfer.
          </p>
          <div className="mt-[26px] flex flex-col gap-3.5">
            {STEPS.map(([n, title, body]) => (
              <div key={n} className="flex gap-3">
                <span className="mono pt-0.5 text-xs text-accent">{n}</span>
                <div>
                  <div className="text-sm" style={{ fontFamily: 'var(--font-heading)' }}>{title}</div>
                  <div className="text-[13px]" style={{ color: MUTED(62) }}>{body}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Real distribution of the live window, in place of the mock's invented SHAP panel */}
        <div className="card elev-sm rounded-lg p-5">
          <div className="flex items-baseline justify-between">
            <div className="text-[15px]" style={{ fontFamily: 'var(--font-heading)' }}>Scores in the live window</div>
            <span className="mono text-[11px]" style={{ color: MUTED(45) }}>{alerts.length.toLocaleString()} transfers</span>
          </div>
          <div className="mt-5 flex flex-col gap-[13px]">
            {bands.map((b) => (
              <div key={b.label}>
                <div className="mb-[5px] flex justify-between text-xs">
                  <span style={{ color: MUTED(78) }}>{b.label} <span className="mono" style={{ color: MUTED(45) }}>{b.hint}</span></span>
                  <span className="mono" style={{ color: b.color }}>{b.n.toLocaleString()}</span>
                </div>
                <div className="h-[5px] rounded-[3px]" style={{ background: MUTED(8) }}>
                  <div className="h-[5px] rounded-[3px]" style={{ background: b.color, width: b.w }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-5 mb-0 text-[12.5px]" style={{ color: MUTED(55) }}>
            Counted from the last 24 hours of scored transfers, refreshed as blocks land.
          </p>
        </div>
      </section>

      {/* ── Pillars ──────────────────────────────────────────────────── */}
      <section id="pillars" className="grid gap-[18px] pt-[88px] md:grid-cols-3">
        {PILLARS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="card elev-sm rounded-lg p-5">
            <Icon size={20} className="text-accent" />
            <div className="mt-3.5 text-base" style={{ fontFamily: 'var(--font-heading)' }}>{title}</div>
            <p className="mb-0 mt-2 text-[13px] leading-[1.6]" style={{ color: MUTED(65) }}>{body}</p>
          </div>
        ))}
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────── */}
      <section className="pt-[88px]">
        <div className="card elev-sm flex-row flex-wrap items-center gap-8 rounded-lg p-8">
          <div className="min-w-[280px] flex-1">
            <h3 className="tracking-[-0.02em]">Check an address before you sign.</h3>
            <p className="mb-0 mt-2.5 text-sm" style={{ color: MUTED(65) }}>
              No wallet, no account. The scanner is read-only and open.
            </p>
          </div>
          <button className="btn btn-primary px-5 py-[11px]" onClick={onEnter}>Launch app<ArrowRight size={14} /></button>
        </div>
        <div className="mt-[26px] flex flex-wrap gap-[18px] text-xs" style={{ color: MUTED(45) }}>
          <span>Recon · Ethereum threat intelligence</span>
          <span className="sm:ml-auto">Scores are probabilistic. Not financial advice.</span>
        </div>
      </section>
    </div>
  )
}
