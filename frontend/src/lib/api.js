// Single place the frontend talks to the backend. Contract is fixed by backend/app/api/api.py.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// GET /api/get-alerts -> [{ id, address, to_address, value, timestamp, tx_hash, probability }]
// The endpoint defaults to limit=200; that caps the "Scanned (24h)" KPI, so ask for the
// full 24h window. ponytail: 5000 ceiling — plenty for this tool's volume; raise if it ever caps.
export async function getAlerts(limit = 5000) {
  const res = await fetch(`${API_URL}/api/get-alerts?limit=${limit}`)
  if (!res.ok) throw new Error(`get-alerts failed: ${res.status}`)
  return res.json()
}

// POST /api/contract-analyzer -> { risk_level, safe_score, summary, contract_name, vulnerabilities[] }
export async function analyzeContract(address) {
  const res = await fetch(`${API_URL}/api/contract-analyzer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ address }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Analysis failed: ${res.status}`)
  }
  return res.json()
}
