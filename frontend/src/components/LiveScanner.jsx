import { useMemo, useState } from 'react'
import { Search, Activity } from 'lucide-react'
import Card from './Card'
import RiskBadge from './RiskBadge'
import AddressPill from './AddressPill'
import EmptyState from './EmptyState'
import { shortHash, formatEth, relativeTime, etherscanTx } from '../lib/format'

export default function LiveScanner({ transactions }) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('latest') // 'latest' | 'risk'

  const rows = useMemo(() => {
    const q = query.toLowerCase().trim()
    let list = transactions
    if (q) {
      list = list.filter(
        (t) =>
          t.tx_hash?.toLowerCase().includes(q) ||
          t.address?.toLowerCase().includes(q) ||
          t.to_address?.toLowerCase().includes(q),
      )
    }
    if (sort === 'risk') {
      list = [...list].sort((a, b) => (b.probability || 0) - (a.probability || 0))
    }
    return list.slice(0, 100)
  }, [transactions, query, sort])

  const actions = (
    <div className="flex items-center gap-2">
      <div className="relative">
        <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter hash / address"
          className="w-40 rounded-lg border border-line bg-app py-1.5 pl-8 pr-2 font-mono text-xs text-ink placeholder:text-muted focus:border-accent focus:outline-none sm:w-56"
        />
      </div>
      <select
        value={sort}
        onChange={(e) => setSort(e.target.value)}
        aria-label="Sort transactions"
        className="rounded-lg border border-line bg-app px-2 py-1.5 text-xs text-ink focus:border-accent focus:outline-none"
      >
        <option value="latest">Latest</option>
        <option value="risk">Risk</option>
      </select>
    </div>
  )

  return (
    <Card
      title="Network Traffic"
      subtitle={`${transactions.length} scanned in the last 24h`}
      icon={Activity}
      actions={actions}
      bodyClassName="p-0"
      className="h-full"
    >
      {rows.length === 0 ? (
        <EmptyState
          icon={Activity}
          title={query ? 'No matching transactions' : 'Listening for blocks'}
          description={query ? 'Try a different hash or address.' : 'New Ethereum transactions will appear here as they are scored.'}
        />
      ) : (
        <div className="h-full overflow-auto">
          <table className="w-full min-w-[720px] border-collapse text-sm">
            <thead className="sticky top-0 z-10 bg-surface">
              <tr className="border-b border-line text-left text-xs font-medium text-muted">
                <th className="px-4 py-2.5">Elapsed</th>
                <th className="px-4 py-2.5">Transaction / Source</th>
                <th className="px-4 py-2.5">Destination</th>
                <th className="px-4 py-2.5 text-right">Value</th>
                <th className="px-4 py-2.5 text-right">Risk</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((tx, i) => (
                <tr
                  key={tx.tx_hash}
                  className={`border-b border-line/60 transition-colors hover:bg-elevated/60 ${i === 0 && sort === 'latest' ? 'row-flash' : ''}`}
                >
                  <td className="whitespace-nowrap px-4 py-3 text-xs text-muted">{relativeTime(tx.timestamp)}</td>
                  <td className="px-4 py-3">
                    <a
                      href={etherscanTx(tx.tx_hash)}
                      target="_blank"
                      rel="noreferrer"
                      className="font-mono text-xs text-accent hover:underline"
                    >
                      {shortHash(tx.tx_hash)}
                    </a>
                    <div className="mt-0.5">
                      <AddressPill address={tx.address} />
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {tx.to_address ? <AddressPill address={tx.to_address} /> : <span className="text-xs text-muted">Contract creation</span>}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right font-mono text-xs tabular">{formatEth(tx.value)} ETH</td>
                  <td className="px-4 py-3 text-right">
                    <RiskBadge probability={tx.probability} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
