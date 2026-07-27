import { useEffect, useMemo, useState } from 'react'
import {
  FileSearch, Loader2, TriangleAlert, RotateCcw, ArrowUpRight,
  RefreshCw, Fish, Crown, Timer, Link2, Percent, Layers, Code,
} from 'lucide-react'
import SafetyScoreRing from './SafetyScoreRing'
import FindingRow from './FindingRow'
import { analyzeContract } from '../lib/api'
import { getScoreBand, severityColor } from '../lib/risk'
import { etherscanAddr } from '../lib/format'
import { ADDRESS_RE } from '../hooks/useWatchlist'

const MUTED = (pct) => `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`
const LABEL = 'text-[11px] uppercase tracking-[0.07em]'
const RECENT_KEY = 'recon-recent-audits'

const EXAMPLES = [
  { label: '0xdAC1…31ec7 · USDT', address: '0xdAC17F958D2ee523a2206206994597C13D831ec7' },
  { label: '0x7a25…488D · UniswapV2Router', address: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D' },
  { label: '0x1526…4c898 · CakeOFT', address: '0x152649eA73beAb28c5b49B26eb48f7EAD6d4c898' },
]

const CHECKS = [
  [RefreshCw, 'Reentrancy', 'External calls before state writes, cross-function and read-only paths.'],
  [Fish, 'Honeypots', 'Asymmetric transfer logic, hidden blocklists, sell-side reverts.'],
  [Crown, 'Owner privilege', 'Pause, mint, cap, fee and whitelist powers held by a single key.'],
  [Timer, 'Timelocks', 'Whether critical setters can execute without notice.'],
  [Link2, 'Cross-chain trust', 'Bridge remotes, message verification, replay surface.'],
  [Percent, 'Fee traps', 'Fee ceilings, denominators and post-deploy fee mutability.'],
  [Layers, 'Proxy safety', 'Upgrade authority, storage-collision risk, uninitialised slots.'],
  [Code, 'Compiler risk', 'Known pragma bugs, unchecked math, deprecated opcodes.'],
]

const SEVERITIES = ['Critical', 'High', 'Medium', 'Low']

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

  // Arriving from "Audit counterparty contract" prefills the field; the audit itself
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

  const band = report ? getScoreBand(report.safe_score) : null
  const findings = report?.vulnerabilities || []

  return (
    <div className="mx-auto flex max-w-[1080px] flex-col gap-3.5">
      <form className="card elev-sm rounded-md p-4" onSubmit={run}>
        <div className="flex flex-wrap gap-[9px]">
          <input
            className="input mono min-h-[38px] flex-1 text-[13px] sm:min-w-[280px]"
            style={{ background: 'var(--color-bg)' }}
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="0x… verified contract address"
            aria-label="Contract address"
          />
          <button type="submit" className="btn btn-primary min-h-[38px] px-[18px]" disabled={loading || !ADDRESS_RE.test(address.trim())}>
            {loading ? <Loader2 size={15} className="rc-spin" /> : <FileSearch size={15} />}
            {loading ? 'Analysing…' : 'Audit'}
          </button>
        </div>
        <div className="mt-[11px] flex flex-wrap items-center gap-2">
          <span className={LABEL} style={{ color: MUTED(45) }}>Try</span>
          {EXAMPLES.map((e) => (
            <button
              key={e.address}
              type="button"
              onClick={() => setAddress(e.address)}
              title="Fill the field — press Audit to run it"
              className="rc-chip mono cursor-pointer rounded-full border bg-transparent px-[9px] py-1 text-[11.5px]"
              style={{ borderColor: 'var(--color-divider)', color: MUTED(70) }}
            >
              {e.label}
            </button>
          ))}
        </div>
        {error && (
          <div
            className="mt-3 flex items-center gap-2 rounded-md border px-3 py-2 text-[13px]"
            style={{ borderColor: 'var(--risk-high)', color: 'var(--risk-high)' }}
            role="alert"
          >
            <TriangleAlert size={15} className="shrink-0" />
            {error}
          </div>
        )}
      </form>

      {loading && (
        <div className="card elev-sm flex-row items-center gap-2.5 rounded-md p-4 text-[13px]">
          <Loader2 size={15} className="rc-spin text-accent" />
          Fetching verified source from Etherscan and running the security review — this takes a few seconds.
        </div>
      )}

      {report && !loading && (
        <div className="grid items-start gap-3.5 lg:grid-cols-[minmax(0,1fr)_292px]">
          <div className="flex min-w-0 flex-col gap-3.5">
            <div className="card elev-sm rounded-md p-[18px]">
              <div className="flex flex-wrap items-start gap-5">
                <SafetyScoreRing score={report.safe_score} />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-[9px]">
                    <h4>{report.contract_name || 'Contract'}</h4>
                    {report.risk_level && (
                      <span className="tag border" style={{ borderColor: band.color, color: band.color }}>
                        {report.risk_level} risk
                      </span>
                    )}
                  </div>
                  <div className="mono mt-1.5 break-all text-[11.5px]" style={{ color: MUTED(50) }}>{report.address}</div>
                  {report.summary && (
                    <p className="mb-0 mt-2.5 text-[13.5px] leading-[1.6] text-pretty" style={{ color: MUTED(75) }}>
                      {report.summary}
                    </p>
                  )}
                </div>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-[9px] sm:grid-cols-4">
                {counts.map((c) => (
                  <div
                    key={c.k}
                    className="rounded-sm px-[11px] py-[9px]"
                    style={{ background: 'var(--surface-2)', boxShadow: `inset 3px 0 0 ${c.color}` }}
                  >
                    <div className="mono text-[17px]" style={{ color: c.color }}>{c.n}</div>
                    <div className="mt-0.5 text-[10.5px] uppercase tracking-[0.06em]" style={{ color: MUTED(50) }}>{c.k}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card elev-sm overflow-hidden rounded-md p-0">
              <div className="flex items-center gap-2 border-b border-line px-[15px] py-3">
                <TriangleAlert size={15} style={{ color: 'var(--risk-med)' }} />
                <span className="text-sm" style={{ fontFamily: 'var(--font-heading)' }}>Findings</span>
                <span className="mono text-[11.5px]" style={{ color: MUTED(45) }}>{findings.length}</span>
                {findings.length > 0 && (
                  <button
                    className="btn btn-ghost ml-auto px-1.5 py-0.5 text-[11.5px]"
                    onClick={() => { setAllOpen((v) => !v); setOpen({}) }}
                  >
                    {allOpen ? 'Collapse all' : 'Expand all'}
                  </button>
                )}
              </div>
              {findings.length === 0 ? (
                <div className="px-[15px] py-8 text-center text-[13px]" style={{ color: MUTED(60) }}>
                  No vulnerabilities were flagged by the analysis.
                </div>
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
            </div>
          </div>

          <div className="flex flex-col gap-3.5">
            <div className="card elev-sm rounded-md p-3.5">
              <div className={LABEL} style={{ color: MUTED(55) }}>This report</div>
              <div className="mt-[11px] flex flex-col gap-[9px] text-xs">
                <div className="flex items-center gap-2">
                  <span style={{ color: MUTED(55) }}>Safety score</span>
                  <span className="mono ml-auto" style={{ color: band.color }}>{report.safe_score} / 100 · {band.label}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span style={{ color: MUTED(55) }}>Findings</span>
                  <span className="mono ml-auto">{findings.length}</span>
                </div>
                {counts.filter((c) => c.n > 0).map((c) => (
                  <div key={c.k} className="flex items-center gap-2">
                    <span style={{ color: MUTED(55) }}>{c.k}</span>
                    <span className="mono ml-auto" style={{ color: c.color }}>{c.n}</span>
                  </div>
                ))}
                <a
                  href={etherscanAddr(report.address)}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-0.5 inline-flex items-center gap-1 no-underline"
                >
                  View source on Etherscan<ArrowUpRight size={11} />
                </a>
              </div>
            </div>

            <RecentAudits recent={recent} onPick={setAddress} />

            <button className="btn btn-secondary btn-block text-[12.5px]" onClick={reset}>
              <RotateCcw size={14} />Audit another contract
            </button>
          </div>
        </div>
      )}

      {!report && !loading && (
        <div className="grid items-start gap-3.5 lg:grid-cols-[minmax(0,1fr)_292px]">
          <div className="card elev-sm rounded-md p-[18px]">
            <div className="text-[15px]" style={{ fontFamily: 'var(--font-heading)' }}>What the review looks for</div>
            <p className="mb-0 mt-1.5 text-[13px]" style={{ color: MUTED(62) }}>
              Verified source is pulled from Etherscan and put through an LLM security review that returns a
              safety score, a risk level and findings by severity.
            </p>
            <div className="mt-4 grid gap-2.5 sm:grid-cols-2">
              {CHECKS.map(([Icon, title, body]) => (
                <div key={title} className="flex gap-2.5 rounded-sm px-3 py-[11px]" style={{ background: 'var(--surface-2)' }}>
                  <Icon size={16} className="flex-none text-accent" />
                  <div className="min-w-0">
                    <div className="text-[13px]" style={{ fontFamily: 'var(--font-heading)' }}>{title}</div>
                    <div className="mt-0.5 text-[11.5px] leading-[1.5]" style={{ color: MUTED(55) }}>{body}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <RecentAudits recent={recent} onPick={setAddress} />
        </div>
      )}
    </div>
  )
}

// Audits run from this browser. Clicking one refills the field rather than re-running it.
function RecentAudits({ recent, onPick }) {
  return (
    <div className="card elev-sm overflow-hidden rounded-md p-0">
      <div className={`${LABEL} border-b border-line px-3.5 py-3`} style={{ color: MUTED(55) }}>Recent audits</div>
      {recent.length === 0 ? (
        <div className="px-3.5 py-6 text-center text-[12.5px]" style={{ color: MUTED(50) }}>
          Reports you run here will be listed for quick recall.
        </div>
      ) : (
        recent.map((a) => {
          const color = getScoreBand(a.score).color
          return (
            <button
              key={a.address}
              onClick={() => onPick(a.address)}
              className="rc-row block w-full cursor-pointer border-b border-0 bg-transparent px-3.5 py-[11px] text-left"
              style={{ boxShadow: `inset 3px 0 0 ${color}`, borderColor: MUTED(6), color: 'inherit', font: 'inherit' }}
            >
              <div className="flex items-center gap-2">
                <span className="truncate text-[13px]">{a.name}</span>
                <span className="mono ml-auto text-xs" style={{ color }}>{a.score}</span>
              </div>
              <div className="mono mt-[3px] truncate text-[11px]" style={{ color: MUTED(45) }}>{a.address}</div>
            </button>
          )
        })
      )}
    </div>
  )
}
