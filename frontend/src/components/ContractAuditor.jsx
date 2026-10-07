import { useEffect, useMemo, useState } from 'react'
import { Loader2, TriangleAlert, ArrowUpRight } from 'lucide-react'
import FindingRow from './FindingRow'
import { analyzeContract } from '../lib/api'
import { severityColor } from '../lib/risk'
import { etherscanAddr } from '../lib/format'
import { ADDRESS_RE } from '../hooks/useWatchlist'

const RECENT_KEY = 'recon-recent-audits'

const EXAMPLES = [
  { label: 'USDT', address: '0xdAC17F958D2ee523a2206206994597C13D831ec7' },
  { label: 'Uniswap V2 Router', address: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D' },
  { label: 'CakeOFT', address: '0x152649eA73beAb28c5b49B26eb48f7EAD6d4c898' },
]

// Exactly what the review prompt asks for (backend/app/services/contract_analyzer.py).
const CHECKS = [
  ['Reentrancy', 'External calls that can re-enter before state is updated.'],
  ['Integer overflow and underflow', 'Unchecked arithmetic where SafeMath or checked math is missing.'],
  ['Unchecked return values', 'Calls and transfers whose failure is silently ignored.'],
  ['Centralisation', 'Owner powers such as unlimited minting.'],
  ['Honeypots', 'Restrictions that stop holders from selling.'],
]

// The finding severity enum the backend returns.
const SEVERITIES = ['High', 'Medium', 'Low']
const SCORE_TICKS = [0, 70, 100] // ≥70 is the shaded reference interval

function loadRecent() {
  try {
    const raw = JSON.parse(localStorage.getItem(RECENT_KEY))
    return Array.isArray(raw) ? raw : []
  } catch {
    return []
  }
}

export default function ContractAuditor({ initialAddress = '' }) {
  const [address, setAddress] = useState(initialAddress)
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')
  const [open, setOpen] = useState({})
  const [allOpen, setAllOpen] = useState(false)
  const [recent, setRecent] = useState(loadRecent)

  // Arriving from "Audit the recipient contract" prefills the field; the audit itself
  // stays a deliberate click — the backend allows only 3 uncached analyses per day.
  useEffect(() => {
    if (initialAddress) setAddress(initialAddress)
  }, [initialAddress])

  const run = async (e) => {
    e.preventDefault()
    const target = address.trim()
    if (!ADDRESS_RE.test(target) || loading) return
    setLoading(true)
    setError('')
    setReport(null)
    setOpen({})
    setAllOpen(false)
    try {
      const result = await analyzeContract(target)
      setReport({ ...result, address: target })
      setRecent((r) => {
        const next = [
          { address: target, name: result.contract_name || 'Contract', score: result.safe_score },
          ...r.filter((a) => a.address.toLowerCase() !== target.toLowerCase()),
        ].slice(0, 4)
        try {
          localStorage.setItem(RECENT_KEY, JSON.stringify(next))
        } catch {
          /* ignore private-mode storage errors */
        }
        return next
      })
    } catch (err) {
      setError(err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setReport(null)
    setAddress('')
    setError('')
    setOpen({})
    setAllOpen(false)
  }

  const counts = useMemo(() => {
    const list = report?.vulnerabilities || []
    return SEVERITIES.map((k) => ({
      k,
      n: list.filter((v) => (v.severity || '').toLowerCase() === k.toLowerCase()).length,
      color: severityColor(k),
    }))
  }, [report])

  const verdictColor = report ? severityColor(report.risk_level) : null
  const findings = report?.vulnerabilities || []
  const valid = ADDRESS_RE.test(address.trim())
  const score = Math.max(0, Math.min(100, report?.safe_score || 0))

  return (
    <div className="flex flex-col gap-4">
      <form className="sheet px-4 py-4 sm:px-5" onSubmit={run}>
        <label htmlFor="audit-address" className="th">Verified contract address</label>
        <div className="mt-1.5 flex flex-col gap-2 sm:flex-row">
          <input
            id="audit-address"
            className="input mono min-h-[40px] flex-1 text-[14px]"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="0x…"
            spellCheck={false}
            autoComplete="off"
          />
          <button type="submit" className="btn btn-primary min-h-[40px] px-5" disabled={loading || !valid}>
            {loading && <Loader2 size={15} className="spin" />}
            {loading ? 'Auditing…' : 'Run audit'}
          </button>
        </div>
        <div className="mt-2.5 flex flex-wrap items-center gap-x-1 gap-y-1 text-[13px]">
          <span className="mr-1 text-ink-3">Examples:</span>
          {EXAMPLES.map((e, i) => (
            <span key={e.address} className="inline-flex items-center">
              <button
                type="button"
                onClick={() => setAddress(e.address)}
                title={`Fill in ${e.address}`}
                className="cursor-pointer bg-transparent p-0 font-semibold text-act underline decoration-1 underline-offset-[3px] hover:text-ink"
              >
                {e.label}
              </button>
              {i < EXAMPLES.length - 1 && <span className="ml-1 text-ink-3">·</span>}
            </span>
          ))}
        </div>
        <p className="mt-2.5 max-w-[78ch] text-[12.5px] text-ink-3">
          New audits are limited to three a day across all users. Contracts audited before return instantly.
        </p>
        {error && (
          <div className="mt-3 flex items-start gap-2 border-t border-line pt-3 text-[13.5px] font-semibold text-risk-high" role="alert">
            <TriangleAlert size={16} className="mt-0.5 shrink-0" />
            {error}
          </div>
        )}
      </form>

      {loading && (
        <div className="sheet flex items-center gap-3 px-5 py-4 text-[13.5px]" role="status">
          <Loader2 size={16} className="spin text-act" />
          Fetching verified source from Etherscan and running the review. This usually takes several seconds.
        </div>
      )}

      {report && !loading && (
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
          <article className="sheet min-w-0">
            <header className="px-5 pb-4 pt-5">
              <h2 className="text-[24px]">{report.contract_name || 'Contract'}</h2>
              <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="mono break-all text-[13px] text-ink-2">{report.address}</span>
                <a href={etherscanAddr(report.address)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-[12.5px] font-semibold no-underline">
                  Source on Etherscan<ArrowUpRight size={12} />
                </a>
              </div>
            </header>

            <div className="border-t border-line-strong px-5 pb-1 pt-4">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="th">Safety score</span>
                <span className="mono text-[28px] font-semibold leading-none" style={{ color: verdictColor }}>{report.safe_score}</span>
                <span className="mono text-[13px] text-ink-3">/ 100</span>
                {report.risk_level && (
                  <span className="text-[14px] font-bold" style={{ color: verdictColor }}>
                    {report.risk_level} risk
                  </span>
                )}
              </div>
              <div className="rb mt-3" style={{ height: 22 }} role="img" aria-label={`Safety score ${report.safe_score} of 100; 70 and above is the shaded range`}>
                <span className="rb-ref" style={{ left: '70%', width: '30%' }} />
                <span className="rb-tick" style={{ left: '70%' }} />
                <span className="rb-mark" style={{ left: `${score}%`, background: verdictColor }} />
              </div>
              <div className="mono relative mt-1.5 h-4 text-[11px] text-ink-3" aria-hidden="true">
                {SCORE_TICKS.map((t) => (
                  <span key={t} className={`absolute ${t === 0 ? '' : t === 100 ? '-translate-x-full' : '-translate-x-1/2'}`} style={{ left: `${t}%` }}>{t}</span>
                ))}
              </div>
            </div>

            {report.summary && (
              <section className="px-5 pt-4">
                <h3 className="text-[15px]">Summary</h3>
                <p className="mt-1.5 max-w-[72ch] text-[14px] leading-[1.6] text-ink-2 text-pretty">{report.summary}</p>
              </section>
            )}

            <section className="mt-5">
              <div className="flex items-center gap-2 border-b border-line-strong px-5 pb-2">
                <h3 className="text-[15px]">Findings</h3>
                <span className="text-[13px] text-ink-3">
                  <span className="mono">{findings.length}</span>
                  {counts.filter((c) => c.n).map((c) => (
                    <span key={c.k}> · <span className="mono font-semibold" style={{ color: c.color }}>{c.n}</span> {c.k.toLowerCase()}</span>
                  ))}
                </span>
                {findings.length > 0 && (
                  <button className="btn btn-ghost ml-auto text-[12.5px]" onClick={() => { setAllOpen((v) => !v); setOpen({}) }}>
                    {allOpen ? 'Collapse all' : 'Expand all'}
                  </button>
                )}
              </div>
              {findings.length === 0 ? (
                <p className="px-5 py-8 text-center text-[13.5px] text-ink-3">The review flagged no vulnerabilities.</p>
              ) : (
                findings.map((v, i) => (
                  <FindingRow
                    key={i}
                    finding={v}
                    open={allOpen || !!open[i]}
                    onToggle={() => { setOpen((o) => ({ ...o, [i]: !(allOpen || o[i]) })); setAllOpen(false) }}
                  />
                ))
              )}
            </section>
          </article>

          <div className="flex flex-col gap-4">
            <button className="btn btn-secondary w-full" onClick={reset}>Audit another contract</button>
            <RecentAudits recent={recent} onPick={setAddress} />
          </div>
        </div>
      )}

      {!report && !loading && (
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
          <section className="sheet">
            <div className="sheet-head">
              <h2 className="sheet-title">What the review checks</h2>
            </div>
            <dl className="m-0">
              {CHECKS.map(([title, body]) => (
                <div key={title} className="grid gap-x-4 border-t border-line px-4 py-2.5 sm:grid-cols-[220px_minmax(0,1fr)]">
                  <dt className="text-[13.5px] font-bold">{title}</dt>
                  <dd className="m-0 text-[13.5px] text-ink-2">{body}</dd>
                </div>
              ))}
            </dl>
          </section>
          <RecentAudits recent={recent} onPick={setAddress} />
        </div>
      )}
    </div>
  )
}

// Audits run from this browser. Clicking one refills the field rather than re-running it.
function RecentAudits({ recent, onPick }) {
  return (
    <section className="sheet" aria-label="Recent audits">
      <div className="sheet-head">
        <h2 className="sheet-title">Recent audits</h2>
        <span className="ml-auto text-[12px] text-ink-3">this browser</span>
      </div>
      {recent.length === 0 ? (
        <p className="px-4 py-6 text-center text-[13px] text-ink-3">None yet.</p>
      ) : (
        recent.map((a) => (
            <button
              key={a.address}
              onClick={() => onPick(a.address)}
              title="Fill the field with this address"
              className="row-btn grid-cols-[minmax(0,1fr)_auto] gap-x-3 px-4 py-2.5"
            >
              <span className="truncate text-[13.5px] font-semibold">{a.name}</span>
              <span className="mono text-[13.5px] font-semibold">{a.score}</span>
              <span className="mono col-span-2 truncate text-[12px] text-ink-3">{a.address}</span>
            </button>
        ))
      )}
    </section>
  )
}
