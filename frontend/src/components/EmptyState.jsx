// Neutral empty/placeholder state used across pages.
export default function EmptyState({ title, description, className = 'px-5 py-12' }) {
  return (
    <div className={`text-center ${className}`}>
      <div className="text-[14px] font-semibold">{title}</div>
      {description && <div className="mx-auto mt-1 max-w-[42ch] text-[13px] text-ink-3">{description}</div>}
    </div>
  )
}
