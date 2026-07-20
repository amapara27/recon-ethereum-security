import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle({ theme, onToggle }) {
  const isDark = theme === 'dark'
  return (
    <button
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="grid size-9 place-items-center rounded-lg border border-line bg-surface text-muted transition-colors hover:text-ink"
    >
      {isDark ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  )
}
