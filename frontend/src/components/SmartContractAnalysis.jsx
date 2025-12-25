import { useState, useEffect } from 'react'
import api from '../api'

// SVG Icons
function ClipboardIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  )
}

function ClearIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

function ChevronIcon({ expanded }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{
        transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
        transition: 'transform 0.2s ease'
      }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  )
}

function ShieldCheckIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  )
}

// Security Score Ring Component
function SecurityScoreRing({ score }) {
  const radius = 54
  const strokeWidth = 8
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (score / 100) * circumference

  const getScoreColor = () => {
    if (score >= 70) return { stroke: '#00ff88', glow: 'rgba(0, 255, 136, 0.4)' }
    if (score >= 40) return { stroke: '#ffa502', glow: 'rgba(255, 165, 2, 0.4)' }
    return { stroke: '#ff4757', glow: 'rgba(255, 71, 87, 0.4)' }
  }

  const colors = getScoreColor()

  return (
    <div className="security-score-ring">
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* Background ring */}
        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
        />
        {/* Progress ring */}
        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke={colors.stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            filter: `drop-shadow(0 0 8px ${colors.glow})`,
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%',
            transition: 'stroke-dashoffset 0.8s ease-out'
          }}
        />
      </svg>
      <div className="security-score-value" style={{ color: colors.stroke }}>
        <span className="score-number">{score}</span>
        <span className="score-label">/ 100</span>
      </div>
    </div>
  )
}

function SmartContractAnalysis() {
  const [contractAddress, setContractAddress] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [loadingText, setLoadingText] = useState('Initializing analysis...')
  const [expandedFindings, setExpandedFindings] = useState(new Set())

  useEffect(() => {
    console.log('SmartContractAnalysis mounted. API base URL:', api.defaults.baseURL)
  }, [])

  useEffect(() => {
    if (!loading) return

    const messages = [
      "🔗 Connecting to Ethereum Network...",
      "📜 Fetching Contract Source Code...",
      "🔍 Scanning for Vulnerabilities...",
      "🧠 AI Agent Analyzing Security Patterns...",
      "⚡ Checking for Reentrancy Attacks...",
      "🛡️ Generating Security Report..."
    ]

    let i = 0
    setLoadingText(messages[0])

    const interval = setInterval(() => {
      i = (i + 1) % messages.length
      setLoadingText(messages[i])
    }, 900)

    return () => clearInterval(interval)
  }, [loading])

  const analyzeContract = async () => {
    if (!contractAddress.trim()) {
      setError('Please enter a contract address')
      return
    }

    if (!/^0x[a-fA-F0-9]{40}$/.test(contractAddress.trim())) {
      setError('Invalid Ethereum address format')
      return
    }

    setLoading(true)
    setError('')
    setAnalysis(null)

    const MIN_LOADING_TIME = 3000;
    const timerPromise = new Promise(resolve => setTimeout(resolve, MIN_LOADING_TIME));

    const apiPromise = api.post('/api/contract-analyzer', {
      address: contractAddress.trim()
    });

    try {
      const [_, response] = await Promise.all([timerPromise, apiPromise]);
      console.log('Analysis response:', response.data);
      setAnalysis(response.data);
    } catch (err) {
      console.error('Analysis error:', err)
      console.error('Error response:', err.response)
      setError(err.response?.data?.detail || err.message || 'Failed to analyze contract. Please check the address and try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      analyzeContract()
    }
  }

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText()
      setContractAddress(text)
    } catch (err) {
      console.error('Failed to read clipboard:', err)
    }
  }

  const handleClear = () => {
    setContractAddress('')
    setAnalysis(null)
    setError('')
  }

  const toggleFinding = (index) => {
    setExpandedFindings(prev => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  const getRiskBadgeClass = (riskLevel) => {
    const level = riskLevel?.toLowerCase()
    if (level === 'critical') return 'severity-critical'
    if (level === 'high') return 'severity-high'
    if (level === 'medium') return 'severity-medium'
    if (level === 'low') return 'severity-low'
    return 'severity-info'
  }

  const getSeverityBadgeClass = (severity) => {
    const sev = severity?.toLowerCase()
    if (sev === 'critical') return 'severity-critical'
    if (sev === 'high') return 'severity-high'
    if (sev === 'medium') return 'severity-medium'
    if (sev === 'low') return 'severity-low'
    return 'severity-info'
  }

  const getSeverityClass = (severity) => {
    const sev = severity?.toLowerCase()
    if (sev === 'critical') return 'finding-critical'
    if (sev === 'high') return 'finding-high'
    if (sev === 'medium') return 'finding-medium'
    if (sev === 'low') return 'finding-low'
    return 'finding-info'
  }

  const getSafetyScoreClass = (score) => {
    if (score >= 70) return 'high'
    if (score >= 40) return 'medium'
    return 'low'
  }

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <h3>AI Smart Contract Analysis</h3>

      <div className="contract-analysis-container">
        <div className="contract-input-group">
          <div className="contract-input-wrapper">
            <input
              type="text"
              value={contractAddress}
              onChange={(e) => setContractAddress(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter contract address (0x...)"
              disabled={loading}
              className="contract-address-input"
            />
            <div className="input-actions">
              {contractAddress ? (
                <button
                  type="button"
                  onClick={handleClear}
                  className="input-action-btn"
                  title="Clear"
                  disabled={loading}
                >
                  <ClearIcon />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handlePaste}
                  className="input-action-btn"
                  title="Paste from clipboard"
                  disabled={loading}
                >
                  <ClipboardIcon />
                </button>
              )}
            </div>
          </div>
          <button
            onClick={analyzeContract}
            disabled={loading}
            className="analyze-btn-gradient"
          >
            {loading ? (
              <>
                <span className="btn-spinner"></span>
                <span>Analyzing</span>
              </>
            ) : (
              <span>Analyze Contract</span>
            )}
          </button>
        </div>
        <p className="input-helper-text">
          Enter an Ethereum contract address to analyze its security and detect potential vulnerabilities
        </p>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {loading && (
        <div className="analysis-loading-state">
          <div className="analysis-loader">
            <div className="loader-ring"></div>
            <div className="loader-ring"></div>
            <div className="loader-ring"></div>
          </div>
          <p className="loading-status-text">{loadingText}</p>
          <div className="loading-progress-bar">
            <div className="loading-progress-fill"></div>
          </div>
        </div>
      )}

      {analysis && !loading && (
        <div className="analysis-results">
          {/* Prominent Security Score */}
          <div className="security-score-hero">
            <SecurityScoreRing score={analysis.safe_score} />
            <div className="security-score-info">
              <h4 className="security-score-title">Security Score</h4>
              <p className={`security-score-status ${getSafetyScoreClass(analysis.safe_score)}`}>
                {analysis.safe_score >= 70 ? 'Low Risk' : analysis.safe_score >= 40 ? 'Medium Risk' : 'High Risk'}
              </p>
              <span className={`auditor-severity-badge ${getRiskBadgeClass(analysis.risk_level)}`}>
                {analysis.risk_level}
              </span>
            </div>
          </div>

          <div className="analysis-header-grid">
            <div className="analysis-stat">
              <div className="analysis-stat-label">Contract Name</div>
              <div className="analysis-stat-value">
                {analysis.contract_name || 'Unknown'}
              </div>
            </div>
            <div className="analysis-stat">
              <div className="analysis-stat-label">Findings</div>
              <div className="analysis-stat-value">
                {analysis.vulnerabilities?.length || 0}
              </div>
            </div>
            {analysis.cached && (
              <div className="analysis-stat">
                <div className="analysis-stat-label">Status</div>
                <span className="auditor-severity-badge severity-info">Cached</span>
              </div>
            )}
          </div>

          <div className="analysis-section">
            <h4>Summary</h4>
            <div className="analysis-summary">
              {analysis.summary}
            </div>
          </div>

          {analysis.vulnerabilities && analysis.vulnerabilities.length > 0 && (
            <div className="analysis-section">
              <h4>Vulnerabilities Found ({analysis.vulnerabilities.length})</h4>
              <div className="findings-list">
                {analysis.vulnerabilities.map((vuln, index) => (
                  <div
                    key={index}
                    className={`finding-card ${getSeverityClass(vuln.severity)} ${expandedFindings.has(index) ? 'expanded' : ''}`}
                  >
                    <button
                      className="finding-header"
                      onClick={() => toggleFinding(index)}
                    >
                      <div className="finding-header-left">
                        <span className={`auditor-severity-badge ${getSeverityBadgeClass(vuln.severity)}`}>
                          {vuln.severity}
                        </span>
                        <span className="finding-type">{vuln.type}</span>
                      </div>
                      <div className="finding-header-right">
                        {vuln.line_number && vuln.line_number !== 'N/A' && (
                          <span className="finding-line">Line {vuln.line_number}</span>
                        )}
                        <ChevronIcon expanded={expandedFindings.has(index)} />
                      </div>
                    </button>
                    <div className={`finding-content ${expandedFindings.has(index) ? 'expanded' : ''}`}>
                      <p className="finding-description">{vuln.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysis.vulnerabilities && analysis.vulnerabilities.length === 0 && (
            <div className="no-vulnerabilities-state">
              <ShieldCheckIcon />
              <div className="no-vulnerabilities-title">All Clear</div>
              <div className="no-vulnerabilities-text">No vulnerabilities detected in this contract</div>
            </div>
          )}

          {analysis.analyzed_at && (
            <div className="analysis-timestamp">
              Analyzed: {new Date(analysis.analyzed_at).toLocaleString()}
            </div>
          )}
        </div>
      )}

      {!analysis && !loading && !error && (
        <div className="empty-state">
          Enter a contract address above to begin security analysis
        </div>
      )}
    </div>
  )
}

export default SmartContractAnalysis
