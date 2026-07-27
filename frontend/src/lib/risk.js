// Single source of truth for turning a fraud probability (0..1) into UI risk semantics.
// Thresholds preserved from the original components (>=0.8 high, >=0.5 medium, else low).
// Colors are returned as CSS custom properties because the design paints risk inline
// (borders, strokes, inset row rules) where a static utility class can't reach.

export const THREAT_THRESHOLD = 0.5 // a tx at/above this is surfaced as an active threat

export const riskColor = (p) =>
  (p || 0) >= 0.8 ? 'var(--risk-high)' : (p || 0) >= THREAT_THRESHOLD ? 'var(--risk-med)' : 'var(--risk-safe)'

export const riskBand = (p) =>
  (p || 0) >= 0.8 ? 'High risk' : (p || 0) >= THREAT_THRESHOLD ? 'Elevated' : 'Low risk'

// Contract auditor: map a 0..100 safety score to a band.
export function getScoreBand(score) {
  const s = score || 0
  if (s >= 90) return { label: 'Safe', color: 'var(--risk-safe)' }
  if (s >= 70) return { label: 'Secure', color: 'var(--risk-safe)' }
  if (s >= 40) return { label: 'Risky', color: 'var(--risk-med)' }
  return { label: 'Vulnerable', color: 'var(--risk-high)' }
}

// Severity coloring for individual findings (string severity from the API).
export function severityColor(severity) {
  const s = (severity || '').toLowerCase()
  if (s === 'critical' || s === 'high') return 'var(--risk-high)'
  if (s === 'medium') return 'var(--risk-med)'
  return 'var(--risk-safe)'
}
