import { ShieldCheck, ArrowRight, Activity, FileSearch, Radar } from 'lucide-react'
import ThemeToggle from './ThemeToggle'

const STATS = [
  { value: '0.99', label: 'Model ROC-AUC on held-out data' },
  { value: '800+', label: 'Behavioural features per address' },
  { value: '24/7', label: 'Live Ethereum mainnet monitoring' },
]

const FEATURES = [
  { icon: Radar, title: 'Live scanning', body: 'Every new mainnet address is scored by a trained model as blocks arrive.' },
  { icon: Activity, title: 'Risk signals', body: 'Behavioural fingerprints surface suspicious transfers in real time.' },
  { icon: FileSearch, title: 'Contract audits', body: 'AI-driven review of verified Solidity for reentrancy, honeypots and more.' },
]

export default function LandingPage({ onEnter, theme, onToggleTheme }) {
  return (
    <div className="relative min-h-dvh overflow-hidden bg-app text-ink">
      {/* soft accent glow, subtle in both themes */}
      <div
        className="pointer-events-none absolute inset-x-0 top-[-10rem] mx-auto h-[28rem] max-w-4xl rounded-full opacity-60 blur-3xl"
        style={{ background: 'radial-gradient(closest-side, var(--accent-soft), transparent)' }}
        aria-hidden="true"
      />

      <header className="relative mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2.5">
          <span className="grid size-9 place-items-center rounded-xl bg-accent text-accent-fg">
            <ShieldCheck size={20} />
          </span>
          <span className="text-lg font-semibold tracking-tight">Recon</span>
        </div>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </header>

      <main className="relative mx-auto max-w-6xl px-6">
        <section className="mx-auto max-w-3xl pt-16 text-center sm:pt-24">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1 text-xs font-medium text-muted">
            <span className="size-1.5 rounded-full bg-accent" aria-hidden="true" />
            Ethereum threat intelligence
          </span>
          <h1 className="mt-5 text-4xl font-semibold tracking-tight sm:text-6xl">
            Spot on-chain fraud
            <br />
            <span className="text-accent">before it reaches you</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base text-muted sm:text-lg">
            Recon scores live Ethereum activity with a trained machine-learning model and audits smart
            contracts for vulnerabilities — in real time.
          </p>
          <div className="mt-8 flex justify-center">
            <button
              onClick={onEnter}
              className="inline-flex items-center gap-2 rounded-xl bg-accent px-6 py-3 text-sm font-semibold text-accent-fg transition-colors hover:bg-accent-hover"
            >
              Launch app
              <ArrowRight size={16} />
            </button>
          </div>

          <dl className="mx-auto mt-14 grid max-w-2xl grid-cols-3 gap-4">
            {STATS.map((s) => (
              <div key={s.label} className="rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow)]">
                <dt className="font-mono text-2xl font-semibold text-ink tabular sm:text-3xl">{s.value}</dt>
                <dd className="mt-1 text-xs text-muted">{s.label}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="mx-auto grid max-w-4xl gap-4 py-16 sm:grid-cols-3 sm:py-24">
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <div key={title} className="rounded-2xl border border-line bg-surface p-5 shadow-[var(--shadow)]">
              <span className="grid size-10 place-items-center rounded-xl bg-accent-soft text-accent">
                <Icon size={20} />
              </span>
              <h3 className="mt-4 text-sm font-semibold text-ink">{title}</h3>
              <p className="mt-1.5 text-sm text-muted">{body}</p>
            </div>
          ))}
        </section>
      </main>
    </div>
  )
}
