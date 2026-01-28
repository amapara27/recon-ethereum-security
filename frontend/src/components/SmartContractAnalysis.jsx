import { useState } from 'react'
import CyberCard from './CyberCard'

// --- ICONS ---
const AuditorIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="M12 8v4" />
    <path d="M12 16h.01" />
  </svg>
)

const ChevronIcon = ({ expanded }) => (
  <svg
    width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease' }}
  >
    <polyline points="6 9 12 15 18 9" />
  </svg>
)

const WarningIcon = ({ color }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
)

// --- SUB-COMPONENT: Vulnerability Item (Morphing) ---
const VulnerabilityItem = ({ vuln, index }) => {
  const [expanded, setExpanded] = useState(false);

  const getSeverityStyle = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return { color: 'var(--accent-danger)', bg: 'rgba(255, 42, 81, 0.1)' };
      case 'high': return { color: 'var(--accent-danger)', bg: 'rgba(255, 42, 81, 0.05)' };
      case 'medium': return { color: '#ffa500', bg: 'rgba(255, 165, 0, 0.05)' };
      default: return { color: 'var(--accent-success)', bg: 'rgba(0, 255, 157, 0.05)' };
    }
  };

  const style = getSeverityStyle(vuln.severity);

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      style={{
        background: style.bg,
        border: `1px solid ${expanded ? style.color : 'var(--border-subtle)'}`,
        borderRadius: '8px',
        marginBottom: '8px',
        padding: '12px 16px',
        cursor: 'pointer',
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <WarningIcon color={style.color} />
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '14px', color: 'white' }}>{vuln.type}</span>
          <span style={{
            fontSize: '10px',
            background: style.color,
            color: 'black',
            padding: '2px 6px',
            borderRadius: '4px',
            fontWeight: '900',
            textTransform: 'uppercase'
          }}>
            {vuln.severity}
          </span>
        </div>
        <ChevronIcon expanded={expanded} />
      </div>

      <div style={{
        maxHeight: expanded ? '200px' : '0',
        opacity: expanded ? 1 : 0,
        marginTop: expanded ? '12px' : '0',
        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
        fontSize: '13px',
        lineHeight: '1.6',
        color: 'var(--text-secondary)'
      }}>
        <div style={{ marginBottom: '8px' }}>{vuln.description}</div>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: style.color,
          background: 'rgba(0,0,0,0.2)',
          padding: '4px 8px',
          borderRadius: '4px',
          display: 'inline-block'
        }}>
          DETECTED AT: {vuln.line_number || 'N/A'}
        </div>
      </div>
    </div>
  );
};

// --- MAIN COMPONENT ---
function SmartContractAnalysis() {
  const [contractAddress, setContractAddress] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [report, setReport] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const handleAnalyze = async (e) => {
    e.preventDefault()
    if (!contractAddress) return

    setAnalyzing(true)
    setErrorMessage('')
    setReport(null)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/contract-analyzer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address: contractAddress })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis Failed');
      }

      const data = await response.json();
      setReport(data);
    } catch (err) {
      console.error("Auditor API Error:", err);
      setErrorMessage(err.message);
    } finally {
      setAnalyzing(false)
    }
  }

  const getRiskColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'critical':
      case 'high': return 'var(--accent-danger)';
      case 'medium': return '#ffa500';
      case 'low': return 'var(--accent-success)';
      default: return 'var(--text-secondary)';
    }
  };

  const getStatusText = (score) => {
    if (score >= 90) return 'SAFE';
    if (score >= 70) return 'SECURE';
    if (score >= 40) return 'RISKY';
    return 'VULNERABLE';
  }

  return (
    <CyberCard title="CONTRACT FORENSICS" icon={<AuditorIcon />}>
      <div style={{ maxWidth: '900px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h2 style={{ fontSize: '28px', marginBottom: '8px', color: 'white', letterSpacing: '0.1em' }}>NEURAL AUDIT ENGINE</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Deconstruct and analyze smart contract source code for threats.
          </p>
        </div>

        <form onSubmit={handleAnalyze} style={{ display: 'flex', gap: '12px', marginBottom: '40px' }}>
          <input
            type="text"
            value={contractAddress}
            onChange={(e) => setContractAddress(e.target.value)}
            placeholder="ENTER CONTRACT ADDRESS (0x...)"
            style={{
              flex: 1,
              background: 'rgba(0,0,0,0.4)',
              border: '1px solid var(--border-subtle)',
              padding: '18px 24px',
              borderRadius: '12px',
              color: 'white',
              fontFamily: 'var(--font-mono)',
              fontSize: '15px',
              outline: 'none',
              transition: 'border-color 0.3s ease'
            }}
          />
          <button
            type="submit"
            disabled={analyzing}
            style={{
              background: 'var(--accent-purple)',
              color: 'white',
              padding: '0 40px',
              borderRadius: '12px',
              fontWeight: '900',
              fontSize: '13px',
              letterSpacing: '0.15em',
              boxShadow: analyzing ? 'none' : '0 0 25px var(--accent-purple-glow)',
              opacity: analyzing ? 0.6 : 1,
              cursor: analyzing ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s'
            }}
          >
            {analyzing ? 'SCANNING...' : 'PROCESS TARGET'}
          </button>
        </form>

        {errorMessage && (
          <div style={{
            background: 'rgba(255, 42, 81, 0.1)',
            border: '1px solid var(--accent-danger)',
            padding: '16px',
            borderRadius: '8px',
            color: 'var(--accent-danger)',
            fontFamily: 'var(--font-mono)',
            fontSize: '13px',
            marginBottom: '40px'
          }}>
            ERROR: {errorMessage.toUpperCase()}
          </div>
        )}

        {analyzing && (
          <div className="tactical-loader" style={{ animation: 'fadeIn 0.4s ease' }}>
            <div className="scanner-ring" />
            <div className="decoding-text">Analyzing Source Code</div>
            <div style={{
              fontSize: '10px',
              color: 'var(--text-tertiary)',
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em'
            }}>
              Neural Auditor v2.0 // DECRYPTING LOGIC MATRICES
            </div>
          </div>
        )}

        {report && !analyzing && (
          <div className="audit-report" style={{ animation: 'fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1)' }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1.2fr 1fr',
              gap: '24px',
              marginBottom: '32px'
            }}>
              {/* Score & Summary Card */}
              <div style={{
                background: 'rgba(255,255,255,0.03)',
                padding: '32px',
                borderRadius: '16px',
                border: '1px solid var(--border-subtle)',
                position: 'relative'
              }}>
                <div style={{ display: 'flex', gap: '24px', marginBottom: '24px' }}>
                  <div style={{
                    width: '100px',
                    height: '100px',
                    borderRadius: '50%',
                    border: `4px solid ${getRiskColor(report.risk_level)}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '32px',
                    fontWeight: '900',
                    color: getRiskColor(report.risk_level),
                    boxShadow: `0 0 30px ${getRiskColor(report.risk_level)}40`,
                    fontFamily: 'var(--font-display)'
                  }}>
                    {report.safe_score}
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-tertiary)', fontSize: '11px', letterSpacing: '0.1em', marginBottom: '4px' }}>TRUST INDEX</div>
                    <div style={{ fontSize: '24px', fontWeight: '900', color: 'white', letterSpacing: '0.05em' }}>{getStatusText(report.safe_score)}</div>
                    <div style={{ color: getRiskColor(report.risk_level), fontSize: '12px', fontWeight: '800', marginTop: '4px', textTransform: 'uppercase' }}>
                      RISK LEVEL: {report.risk_level}
                    </div>
                  </div>
                </div>

                <h4 style={{ fontSize: '12px', color: 'var(--accent-cyan)', marginBottom: '12px', letterSpacing: '0.1em' }}>SENTIMENT ANALYSIS</h4>
                <p style={{ fontSize: '14px', lineHeight: '1.6', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                  "{report.summary}"
                </p>
                <div style={{ marginTop: '16px', fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                  TARGET: {report.contract_name?.toUpperCase() || 'UNKNOWN SUBSTRATE'}
                </div>
              </div>

              {/* Vulnerabilities List */}
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <h4 style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginBottom: '16px', fontFamily: 'var(--font-display)', letterSpacing: '0.1em' }}>
                  FORENSIC MATRICES ({report.vulnerabilities?.length || 0})
                </h4>
                <div style={{ overflowY: 'auto', maxHeight: '380px', paddingRight: '8px' }}>
                  {report.vulnerabilities && report.vulnerabilities.length > 0 ? (
                    report.vulnerabilities.map((vuln, i) => (
                      <VulnerabilityItem key={i} vuln={vuln} index={i} />
                    ))
                  ) : (
                    <div style={{ textAlign: 'center', padding: '40px', background: 'rgba(0, 255, 157, 0.05)', borderRadius: '12px', border: '1px dashed var(--accent-success)' }}>
                      <div style={{ fontSize: '24px', marginBottom: '8px' }}>✅</div>
                      <div style={{ color: 'var(--accent-success)', fontWeight: 'bold', fontSize: '14px' }}>NO VULNERABILITIES DETECTED</div>
                      <div style={{ color: 'var(--text-tertiary)', fontSize: '11px', marginTop: '4px' }}>CONTRACT STRUCTURE APPEARS ROBUST</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </CyberCard>
  )
}

export default SmartContractAnalysis
