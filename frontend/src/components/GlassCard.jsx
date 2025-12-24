import styles from './GlassCard.module.css'

/**
 * GlassCard - Reusable glassmorphism container component
 *
 * @param {Object} props
 * @param {React.ReactNode} props.children - Content to render inside the card
 * @param {string} [props.className] - Additional CSS classes
 * @param {boolean} [props.interactive] - Whether the card should have interactive hover effects
 * @param {boolean} [props.noPadding] - Remove default padding
 * @param {boolean} [props.compact] - Use compact padding
 * @param {boolean} [props.spacious] - Use spacious padding
 * @param {boolean} [props.animated] - Add fade-in animation
 * @param {'primary'|'success'|'danger'|'warning'} [props.glow] - Add colored glow effect
 * @param {'primary'|'success'|'danger'|'warning'} [props.border] - Add colored border
 * @param {Function} [props.onClick] - Click handler
 * @param {Object} [props.style] - Inline styles
 */
function GlassCard({
  children,
  className = '',
  interactive = false,
  noPadding = false,
  compact = false,
  spacious = false,
  animated = false,
  glow,
  border,
  onClick,
  style,
  ...rest
}) {
  const classes = [
    styles.glassCard,
    interactive && styles.interactive,
    noPadding && styles.noPadding,
    compact && styles.compact,
    spacious && styles.spacious,
    animated && styles.animated,
    glow && styles[`glow${glow.charAt(0).toUpperCase()}${glow.slice(1)}`],
    border && styles[`border${border.charAt(0).toUpperCase()}${border.slice(1)}`],
    className
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={classes}
      onClick={onClick}
      style={style}
      {...rest}
    >
      {children}
    </div>
  )
}

export default GlassCard
