function LandingPage({ onEnter }) {
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#050511',
      backgroundImage: `
        radial-gradient(circle at 50% 50%, rgba(0, 240, 255, 0.1), transparent 40%),
        radial-gradient(circle at 80% 20%, rgba(189, 0, 255, 0.08), transparent 30%)
      `,
      fontFamily: "'Oxanium', sans-serif",
      color: 'white',
      textAlign: 'center'
    }}>
      {/* Logo/Title */}
      <h1 style={{
        fontSize: '72px',
        fontWeight: '700',
        letterSpacing: '0.1em',
        margin: 0,
        background: 'linear-gradient(90deg, #00f0ff, #bd00ff)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        textShadow: '0 0 60px rgba(0, 240, 255, 0.3)'
      }}>
        RECON
      </h1>

      {/* Subtitle */}
      <p style={{
        fontSize: '16px',
        color: '#94a3b8',
        letterSpacing: '0.2em',
        marginTop: '16px',
        fontFamily: "'Inter', sans-serif",
        textTransform: 'uppercase'
      }}>
        Ethereum Fraud Detection System
      </p>

      {/* Enter Button */}
      <button
        onClick={onEnter}
        style={{
          marginTop: '48px',
          padding: '16px 48px',
          background: 'transparent',
          border: '1px solid #00f0ff',
          color: '#00f0ff',
          fontSize: '14px',
          fontWeight: '600',
          letterSpacing: '0.15em',
          cursor: 'pointer',
          fontFamily: "'Oxanium', sans-serif",
          transition: 'all 0.3s ease',
          boxShadow: '0 0 20px rgba(0, 240, 255, 0.2)'
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.background = 'rgba(0, 240, 255, 0.1)';
          e.currentTarget.style.boxShadow = '0 0 30px rgba(0, 240, 255, 0.4)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 240, 255, 0.2)';
        }}
      >
        ENTER DASHBOARD →
      </button>
    </div>
  )
}

export default LandingPage
