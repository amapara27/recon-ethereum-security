import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle({ theme, onToggle, size = 36 }) {
  const isDark = theme === 'dark'
  return (
    <button
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="btn btn-secondary btn-icon"
      style={{ width: size, height: size }}
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  )
}
