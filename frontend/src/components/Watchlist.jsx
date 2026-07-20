import { Eye, Construction } from 'lucide-react'
import Card from './Card'
import EmptyState from './EmptyState'

// Honest placeholder — this feature has no backend yet, so it shows no fabricated data.
export default function Watchlist() {
  return (
    <div className="mx-auto max-w-3xl">
      <Card title="Watchlist" subtitle="Track wallets for behavioural changes" icon={Eye}>
        <EmptyState
          icon={Construction}
          title="Coming soon — not yet live"
          description="Wallet surveillance will let you track specific addresses and get alerted when their transaction behaviour shifts. It is not wired to the backend yet, so there is no live data to show."
        >
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-elevated px-3 py-1 text-xs font-medium text-muted">
            <span className="size-1.5 rounded-full bg-risk-med" aria-hidden="true" />
            On the roadmap
          </span>
        </EmptyState>
      </Card>
    </div>
  )
}
