import { Sun, Moon } from 'lucide-react'

// `onShell` = sitting on the lab-green band, where ink colours don't apply.
export default function ThemeToggle({ theme, onToggle, onShell = false }) {
  const isDark = theme === 'dark'
  return (
    <button
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className={`btn btn-icon ${onShell ? 'h-8 min-h-8 text-on-shell hover:bg-on-shell/10' : 'btn-secondary'}`}
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  )
}
