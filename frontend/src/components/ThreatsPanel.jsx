import { useMemo } from 'react'
import { ShieldAlert, ShieldCheck, ExternalLink } from 'lucide-react'
import Card from './Card'
import RiskBadge from './RiskBadge'
import AddressPill from './AddressPill'
import EmptyState from './EmptyState'
import { THREAT_THRESHOLD } from '../lib/risk'
import { shortHash, etherscanTx } from '../lib/format'

export default function ThreatsPanel({ transactions }) {
  const threats = useMemo(
    () =>
      [...transactions]
        .filter((t) => (t.probability || 0) >= THREAT_THRESHOLD)
        .sort((a, b) => (b.probability || 0) - (a.probability || 0)),
    [transactions],
  )

  return (
    <Card
      title="Active Threats"
      subtitle={threats.length ? `${threats.length} flagged ≥ ${THREAT_THRESHOLD * 100}%` : 'Real-time'}
      icon={ShieldAlert}
      bodyClassName="p-0"
      className="h-full"
    >
      {threats.length === 0 ? (
        <EmptyState icon={ShieldCheck} title="No active threats" description="No transactions above the alert threshold in the current window." />
      ) : (
        <div className="h-full space-y-2 overflow-auto p-3">
          {threats.map((t) => (
            <div key={t.tx_hash} className="rounded-xl border border-risk-high/30 bg-risk-high/5 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-risk-high">High-risk transfer</span>
                <RiskBadge probability={t.probability} />
              </div>
              <div className="mt-2 flex items-center justify-between gap-2">
                <AddressPill address={t.address} />
                <a
                  href={etherscanTx(t.tx_hash)}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 font-mono text-xs text-accent hover:underline"
                >
                  {shortHash(t.tx_hash, 8)}
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
