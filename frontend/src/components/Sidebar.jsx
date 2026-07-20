import { ShieldCheck } from 'lucide-react'

// Desktop / tablet vertical nav (icon rail on md, labelled on lg). Hidden on mobile.
export default function Sidebar({ nav, current, onNavigate }) {
  return (
    <aside className="sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-line bg-surface md:flex md:w-[72px] lg:w-60">
      <div className="flex h-16 items-center gap-2.5 px-4 lg:px-5">
        <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-accent text-accent-fg">
          <ShieldCheck size={20} />
        </span>
        <span className="hidden text-base font-semibold tracking-tight lg:block">Recon</span>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {nav.map(({ id, label, icon: Icon }) => {
          const active = current === id
          return (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              aria-current={active ? 'page' : undefined}
              title={label}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                active ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-elevated hover:text-ink'
              }`}
            >
              <Icon size={19} className="shrink-0" />
              <span className="hidden lg:block">{label}</span>
            </button>
          )
        })}
      </nav>
    </aside>
  )
}

// Mobile bottom tab bar.
export function MobileNav({ nav, current, onNavigate }) {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 flex border-t border-line bg-surface md:hidden">
      {nav.map(({ id, short, icon: Icon }) => {
        const active = current === id
        return (
          <button
            key={id}
            onClick={() => onNavigate(id)}
            aria-current={active ? 'page' : undefined}
            className={`flex flex-1 flex-col items-center gap-1 py-2.5 text-xs font-medium ${
              active ? 'text-accent' : 'text-muted'
            }`}
          >
            <Icon size={20} />
            {short}
          </button>
        )
      })}
    </nav>
  )
}
