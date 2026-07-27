// Neutral empty/placeholder state used across pages.
export default function EmptyState({ icon: Icon, title, description, iconColor, className = 'px-5 py-14' }) {
  return (
    <div className={`text-center ${className}`}>
      {Icon && <Icon size={22} className="mx-auto text-muted" style={iconColor ? { color: iconColor } : undefined} aria-hidden="true" />}
      <div className="mt-2.5 text-[13.5px]">{title}</div>
      {description && <div className="mt-1 text-[12.5px] text-muted">{description}</div>}
    </div>
  )
}
