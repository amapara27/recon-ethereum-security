import { useState } from 'react'
import { FileSearch, Loader2, ShieldCheck, TriangleAlert } from 'lucide-react'
import Card from './Card'
import SafetyScoreRing from './SafetyScoreRing'
import VulnerabilityCard from './VulnerabilityCard'
import EmptyState from './EmptyState'
import { getSeverity } from '../lib/risk'
import { analyzeContract } from '../lib/api'

export default function ContractAuditor() {
  const [address, setAddress] = useState('')
  const [loading, setLoading] = useState(false)
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')

  const run = async (e) => {
    e.preventDefault()
    const target = address.trim()
    if (!target || loading) return
    setLoading(true)
    setError('')
    setReport(null)
    try {
      setReport(await analyzeContract(target))
    } catch (err) {
      setError(err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const sev = report ? getSeverity(report.risk_level) : null

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4 lg:gap-6">
      <Card title="Smart Contract Auditor" subtitle="AI-powered Solidity security analysis" icon={FileSearch}>
        <form onSubmit={run} className="flex flex-col gap-2 sm:flex-row">
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Contract address (0x…)"
            className="flex-1 rounded-lg border border-line bg-app px-3.5 py-2.5 font-mono text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading || !address.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-accent-fg transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <FileSearch size={16} />}
            {loading ? 'Analyzing…' : 'Audit'}
          </button>
        </form>
        {error && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-risk-high/40 bg-risk-high/10 px-3 py-2 text-sm text-risk-high">
            <TriangleAlert size={16} className="shrink-0" />
            {error}
          </div>
        )}
      </Card>

      {loading && (
        <Card>
          <EmptyState icon={Loader2} title="Fetching source & auditing" description="Retrieving verified source from Etherscan and running the security model." />
        </Card>
      )}

      {report && !loading && (
        <Card>
          <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-start">
            <SafetyScoreRing score={report.safe_score} />
            <div className="min-w-0 flex-1 text-center sm:text-left">
              <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start">
                <h3 className="text-lg font-semibold text-ink">{report.contract_name || 'Contract'}</h3>
                {report.risk_level && (
                  <span className={`rounded-md border px-2 py-0.5 text-xs font-semibold uppercase ${sev.text} ${sev.border}`}>
                    {report.risk_level} risk
                  </span>
                )}
              </div>
              {report.summary && <p className="mt-2 text-sm leading-relaxed text-muted">{report.summary}</p>}
            </div>
          </div>

          <div className="mt-6">
            <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink">
              {report.vulnerabilities?.length ? <TriangleAlert size={16} className="text-risk-med" /> : <ShieldCheck size={16} className="text-risk-safe" />}
              Findings {report.vulnerabilities?.length ? `(${report.vulnerabilities.length})` : ''}
            </h4>
            {report.vulnerabilities?.length ? (
              <div className="space-y-2">
                {report.vulnerabilities.map((v, i) => (
                  <VulnerabilityCard key={i} vuln={v} />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted">No vulnerabilities were flagged by the analysis.</p>
            )}
          </div>
        </Card>
      )}

      {!report && !loading && !error && (
        <Card>
          <EmptyState
            icon={FileSearch}
            title="Audit a smart contract"
            description="Enter a verified contract address to run an AI security review — reentrancy, honeypots, access control and more."
          />
        </Card>
      )}
    </div>
  )
}
