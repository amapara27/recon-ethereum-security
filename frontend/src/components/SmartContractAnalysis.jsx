import { useState, useEffect } from 'react'
import api from '../api'

function SmartContractAnalysis() {
  const [contractAddress, setContractAddress] = useState('')
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [loadingText, setLoadingText] = useState('Initializing analysis...')

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

  const getRiskBadgeClass = (riskLevel) => {
    const level = riskLevel?.toLowerCase()
    if (level === 'critical' || level === 'high') return 'bg-danger'
    if (level === 'medium') return 'bg-warning'
    return 'bg-safe'
  }

  const getSeverityBadgeClass = (severity) => {
    const sev = severity?.toLowerCase()
    if (sev === 'high') return 'bg-danger'
    if (sev === 'medium') return 'bg-warning'
    return 'bg-safe'
  }

  const getSeverityClass = (severity) => {
    const sev = severity?.toLowerCase()
    if (sev === 'high') return 'high-severity'
    if (sev === 'medium') return 'medium-severity'
    return 'low-severity'
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
          <input
            type="text"
            value={contractAddress}
            onChange={(e) => setContractAddress(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter contract address (0x...)"
            disabled={loading}
            className="contract-address-input"
          />
          <button
            onClick={analyzeContract}
            disabled={loading}
            className="analyze-btn"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
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
        <div className="empty-state">
          <div className="loading-spinner"></div>
          <p>{loadingText}</p>
        </div>
      )}

      {analysis && !loading && (
        <div className="analysis-results">
          <div className="analysis-header-grid">
            <div className="analysis-stat">
              <div className="analysis-stat-label">Contract Name</div>
              <div className="analysis-stat-value">
                {analysis.contract_name || 'Unknown'}
              </div>
            </div>
            <div className="analysis-stat">
              <div className="analysis-stat-label">Safety Score</div>
              <div className={`safety-score ${getSafetyScoreClass(analysis.safe_score)}`}>
                {analysis.safe_score}/100
              </div>
            </div>
            <div className="analysis-stat">
              <div className="analysis-stat-label">Risk Level</div>
              <span className={`badge ${getRiskBadgeClass(analysis.risk_level)}`}>
                {analysis.risk_level}
              </span>
            </div>
            {analysis.cached && (
              <div className="analysis-stat">
                <div className="analysis-stat-label">Status</div>
                <span className="badge bg-safe">Cached</span>
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
              <div className="vulnerabilities-list">
                {analysis.vulnerabilities.map((vuln, index) => (
                  <div
                    key={index}
                    className={`vulnerability-card ${getSeverityClass(vuln.severity)}`}
                  >
                    <div className="vulnerability-header">
                      <div className="vulnerability-type">
                        {vuln.type}
                      </div>
                      <span className={`badge ${getSeverityBadgeClass(vuln.severity)}`}>
                        {vuln.severity}
                      </span>
                    </div>
                    <div className="vulnerability-description">
                      {vuln.description}
                    </div>
                    {vuln.line_number && vuln.line_number !== 'N/A' && (
                      <div className="vulnerability-line">
                        Line: {vuln.line_number}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysis.vulnerabilities && analysis.vulnerabilities.length === 0 && (
            <div className="no-vulnerabilities">
              <div className="no-vulnerabilities-icon">✓</div>
              <div className="no-vulnerabilities-text">No vulnerabilities detected</div>
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
