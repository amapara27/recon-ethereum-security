// Single source of truth for turning a fraud probability (0..1) into UI risk semantics.
// Thresholds preserved from the original components (>=0.8 high, >=0.5 medium, else low).

export const THREAT_THRESHOLD = 0.5 // a tx at/above this is surfaced as an active threat

export function getRisk(probability) {
  const p = probability || 0
  if (p >= 0.8) {
    return { level: 'high', label: 'High', text: 'text-risk-high', bg: 'bg-risk-high', border: 'border-risk-high' }
  }
  if (p >= THREAT_THRESHOLD) {
    return { level: 'medium', label: 'Elevated', text: 'text-risk-med', bg: 'bg-risk-med', border: 'border-risk-med' }
  }
  return { level: 'low', label: 'Low', text: 'text-risk-safe', bg: 'bg-risk-safe', border: 'border-risk-safe' }
}

// Contract auditor: map a 0..100 safety score to a band. Preserved from SmartContractAnalysis.
export function getScoreBand(score) {
  const s = score || 0
  if (s >= 90) return { label: 'Safe', text: 'text-risk-safe', stroke: 'var(--risk-safe)' }
  if (s >= 70) return { label: 'Secure', text: 'text-risk-safe', stroke: 'var(--risk-safe)' }
  if (s >= 40) return { label: 'Risky', text: 'text-risk-med', stroke: 'var(--risk-med)' }
  return { label: 'Vulnerable', text: 'text-risk-high', stroke: 'var(--risk-high)' }
}

// Severity coloring for individual vulnerabilities (string severity from the API).
export function getSeverity(severity) {
  const s = (severity || '').toLowerCase()
  if (s === 'critical' || s === 'high') return { text: 'text-risk-high', bg: 'bg-risk-high/10', border: 'border-risk-high/40' }
  if (s === 'medium') return { text: 'text-risk-med', bg: 'bg-risk-med/10', border: 'border-risk-med/40' }
  return { text: 'text-risk-safe', bg: 'bg-risk-safe/10', border: 'border-risk-safe/40' }
}
