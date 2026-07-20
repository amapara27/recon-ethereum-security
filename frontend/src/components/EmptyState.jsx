// Neutral empty/placeholder state used across pages.
export default function EmptyState({ icon: Icon, title, description, children }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 py-12 text-center">
      {Icon && (
        <span className="mb-4 grid size-12 place-items-center rounded-xl bg-elevated text-muted">
          <Icon size={24} aria-hidden="true" />
        </span>
      )}
      <p className="text-sm font-semibold text-ink">{title}</p>
      {description && <p className="mt-1 max-w-sm text-sm text-muted">{description}</p>}
      {children && <div className="mt-4">{children}</div>}
    </div>
  )
}
