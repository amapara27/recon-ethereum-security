// Text wordmark plus a range-bar glyph. Single place to swap in a real logo later.
export default function Wordmark({ className = '' }) {
  return (
    <span className={`inline-flex items-center gap-2 text-[17px] font-bold tracking-[-0.02em] ${className}`}>
      <svg width="20" height="14" viewBox="0 0 20 14" aria-hidden="true">
        <rect x="0" y="4" width="10" height="6" fill="currentColor" opacity=".28" />
        <path d="M0 7h20" stroke="currentColor" strokeWidth="1.4" />
        <rect x="13" y="0" width="2.6" height="14" fill="currentColor" />
      </svg>
      Recon
    </span>
  )
}
