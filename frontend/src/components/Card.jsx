// The single surface primitive. Replaces CyberCard + GlassCard.
export default function Card({ title, subtitle, icon: Icon, actions, className = '', bodyClassName = '', children, ...rest }) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-2xl border border-line bg-surface shadow-[var(--shadow)] ${className}`}
      {...rest}
    >
      {(title || actions) && (
        <header className="flex items-center gap-3 border-b border-line px-5 py-4">
          {Icon && (
            <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
              <Icon size={18} aria-hidden="true" />
            </span>
          )}
          <div className="min-w-0 flex-1">
            {title && <h2 className="truncate text-sm font-semibold tracking-tight text-ink">{title}</h2>}
            {subtitle && <p className="truncate text-xs text-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName || 'p-5'}`}>{children}</div>
    </section>
  )
}
