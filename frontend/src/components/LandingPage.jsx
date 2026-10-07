import { useMemo } from 'react'
import { ArrowRight, ArrowUpRight } from 'lucide-react'
import ThemeToggle from './ThemeToggle'
import Wordmark from './Wordmark'
import RangeBar from './RangeBar'
import { shortAddr, formatPct, relativeTime } from '../lib/format'
import { riskColor, riskBand, THREAT_THRESHOLD } from '../lib/risk'
import { indexByAddress, seriesFor } from '../lib/series'

const REPO = 'https://github.com/amapara27/recon-ethereum-security'

const STEPS = [
  ['Ingest', 'The monitor reads each new mainnet block and picks up to five senders it has not scored before. Coverage is partial on purpose: it keeps within free API limits.'],
  ['Fingerprint', 'Each address’s ETH and ERC-20 history from Etherscan becomes 814 behavioural features: cadence, counterparty spread, value distribution, token mix.'],
  ['Score', 'A random forest returns a fraud probability. The transfer and its score go into a feed that keeps the last 24 hours.'],
]

// Numbers documented in the README for the trained classifier.
const MODEL = [
  ['Model', 'Random forest'],
  ['Features per address', '814'],
  ['ROC-AUC, held-out', '0.99'],
  ['Recall, held-out fraud', '0.96'],
]

const LEGEND = [
  ['0–50%', 'Reference interval', 'No flag. Shaded on every bar.', 'var(--ink)'],
  ['50–80%', 'Elevated', 'Flagged for a second look.', 'var(--risk-med)'],
  ['80–100%', 'High', 'Flagged, highest priority.', 'var(--risk-high)'],
]

export default function LandingPage({ onEnter, onAudit, alerts, theme, onToggleTheme }) {
  const { preview, index, example } = useMemo(() => ({
    preview: alerts.slice(0, 7),
    index: indexByAddress(alerts),
    // A real result to annotate: the newest flagged one, else the newest of any kind.
    example: alerts.find((a) => (a.probability || 0) >= THREAT_THRESHOLD) || alerts[0] || null,
  }), [alerts])

  return (
    <div className="min-h-dvh bg-app text-ink">
      <header className="mx-auto flex max-w-[1200px] flex-wrap items-center gap-x-6 gap-y-3 px-5 py-4 sm:px-8">
        <Wordmark className="mr-auto" accent="var(--act)" />
        <nav className="hidden items-center gap-5 text-[14px] font-semibold sm:flex">
          <button onClick={onEnter} className="cursor-pointer bg-transparent text-ink-2 hover:text-ink">Scanner</button>
          <button onClick={onAudit} className="cursor-pointer bg-transparent text-ink-2 hover:text-ink">Auditor</button>
          <a href={REPO} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-ink-2 no-underline hover:text-ink">
            Source<ArrowUpRight size={13} />
          </a>
        </nav>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </header>

      <main className="mx-auto max-w-[1200px] px-5 sm:px-8">
        <section className="grid items-start gap-10 pb-16 pt-8 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] lg:gap-14 lg:pt-16">
          <div className="lg:pt-4">
            <h1 className="max-w-[16ch] text-[38px] font-semibold leading-[1.05] tracking-[-0.03em] sm:text-[50px]">
              Fraud scores for Ethereum transfers, read as blocks land.
            </h1>
            <p className="mt-5 max-w-[48ch] text-[16px] leading-[1.6] text-ink-2 text-pretty">
              Recon fingerprints new mainnet addresses from their on-chain history and scores each transfer with a
              trained classifier. Anything at or above {THREAT_THRESHOLD * 100}% is flagged. A separate auditor
              reviews verified Solidity on request.
            </p>
            <div className="mt-7 flex flex-wrap gap-2.5">
              <button className="btn btn-primary min-h-[40px] px-5" onClick={onEnter}>
                Open the scanner<ArrowRight size={15} />
              </button>
              <button className="btn btn-secondary min-h-[40px] px-5" onClick={onAudit}>Audit a contract</button>
            </div>
          </div>

          {/* The same rows the scanner shows, straight from the API */}
          <section className="sheet" aria-label="Latest results">
            <div className="sheet-head">
              <h2 className="sheet-title">Latest results</h2>
              <span className={`size-[7px] rounded-full ${alerts.length ? 'pulse' : ''}`} style={{ background: alerts.length ? 'var(--act)' : 'var(--ink-3)' }} aria-hidden="true" />
              <span className="ml-auto text-[12.5px] text-ink-3">{alerts.length.toLocaleString()} scored in 24h</span>
            </div>
            {preview.length === 0 && (
              <p className="px-4 py-12 text-center text-[13.5px] text-ink-3">Waiting for the first scored block…</p>
            )}
            {preview.map((t) => {
              const p = t.probability || 0
              return (
                <div key={t.tx_hash} className="mini-grid items-center border-b border-line px-4 py-2.5">
                  <span className="text-[12.5px] text-ink-3" style={{ gridArea: 'age' }}>{relativeTime(t.timestamp)}</span>
                  <span className="mono truncate text-[13px] text-ink-2" style={{ gridArea: 'parties' }}>
                    {shortAddr(t.address)} → {t.to_address ? shortAddr(t.to_address) : 'new contract'}
                  </span>
                  <RangeBar value={p} history={seriesFor(index, t.address)} height={16} style={{ gridArea: 'bar' }} />
                  <span className={`mono text-right text-[13.5px] ${p >= THREAT_THRESHOLD ? 'font-bold' : ''}`} style={{ gridArea: 'pct', color: riskColor(p) }}>
                    {formatPct(p)}
                  </span>
                </div>
              )
            })}
            <button className="btn btn-ghost w-full justify-between rounded-none px-4 py-3 text-[13px]" onClick={onEnter}>
              See every result in the scanner<ArrowRight size={14} />
            </button>
          </section>
        </section>

        <section className="grid gap-10 border-t border-line-strong py-12 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] lg:gap-14">
          <div>
            <h2 className="text-[24px]">How a score is made</h2>
            <dl className="m-0 mt-5">
              {STEPS.map(([title, body]) => (
                <div key={title} className="border-t border-line py-3.5">
                  <dt className="text-[15px] font-bold">{title}</dt>
                  <dd className="m-0 mt-1 max-w-[56ch] text-[14.5px] leading-[1.6] text-ink-2">{body}</dd>
                </div>
              ))}
            </dl>
            <dl className="m-0 mt-4 grid grid-cols-2 border-t border-line-strong">
              {MODEL.map(([k, v]) => (
                <div key={k} className="py-3 pr-3">
                  <dt className="th">{k}</dt>
                  <dd className="m-0 mt-0.5 text-[14px]">{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div>
            <h2 className="text-[24px]">Reading a score</h2>
            <p className="mt-2 max-w-[60ch] text-[14.5px] leading-[1.6] text-ink-2">
              Every score in Recon sits on the same 0–100% scale, so bars compare at a glance. Like a lab result, only
              values outside the reference interval get colour.
            </p>
            {example && (
              <figure className="sheet m-0 mt-5 px-4 pb-3 pt-4">
                <div className="flex items-baseline gap-3">
                  <span className="mono text-[26px] font-semibold leading-none" style={{ color: riskColor(example.probability) }}>
                    {formatPct(example.probability)}
                  </span>
                  <span className="text-[13.5px] font-bold" style={{ color: riskColor(example.probability) }}>{riskBand(example.probability)}</span>
                  <span className="mono ml-auto truncate text-[12.5px] text-ink-3">{shortAddr(example.address)}</span>
                </div>
                <RangeBar value={example.probability} history={seriesFor(index, example.address)} height={22} axis className="mt-3" />
                <figcaption className="mt-1 text-[12.5px] text-ink-3">Hollow dots: the same address’s other scores in the window.</figcaption>
              </figure>
            )}
            <dl className="m-0 mt-5">
              {LEGEND.map(([range, name, body, color]) => (
                <div key={range} className="grid grid-cols-[76px_minmax(0,1fr)] gap-x-4 border-t border-line py-2.5">
                  <dt className="mono text-[13.5px]">{range}</dt>
                  <dd className="m-0 text-[14px]">
                    <span className="font-bold" style={{ color }}>{name}.</span> <span className="text-ink-2">{body}</span>
                  </dd>
                </div>
              ))}
            </dl>

            <div className="mt-8 border-t border-line-strong pt-4">
              <h3 className="text-[15px] font-bold">Contract audits</h3>
              <p className="mt-1 max-w-[60ch] text-[14.5px] leading-[1.6] text-ink-2">
                Separate from the fraud score. Give the auditor a verified contract address and it pulls the Solidity
                from Etherscan, then returns a 0–100 safety score, a risk level and findings by severity with line
                numbers. Contracts audited before come back instantly.
              </p>
              <button className="btn btn-ghost mt-2 -ml-1.5 text-[14px]" onClick={onAudit}>
                Audit a contract<ArrowRight size={14} />
              </button>
            </div>
          </div>
        </section>

        <footer className="flex flex-wrap gap-x-6 gap-y-2 border-t border-line py-6 text-[13px] text-ink-3">
          <a href={REPO} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1">
            Source on GitHub<ArrowUpRight size={12} />
          </a>
        </footer>
      </main>
    </div>
  )
}
