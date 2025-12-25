function LandingPage({ onEnter }) {
  return (
    <div className="landing-container">
      <div className="landing-content">
        <h1 className="landing-title">Recon</h1>
        <p className="landing-subtitle">Ethereum Blockchain Dashboard</p>
        <button className="enter-dashboard-btn" onClick={onEnter}>
          <span>Enter</span>
          <svg className="arrow-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  )
}

export default LandingPage
