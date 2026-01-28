import React from 'react';

const CyberCard = ({ children, title, icon, className = "", variant = "default" }) => {
    const getBorderColor = () => {
        switch (variant) {
            case 'danger': return 'var(--accent-danger)';
            case 'success': return 'var(--accent-success)';
            default: return 'var(--accent-cyan)';
        }
    };

    const cardStyle = {
        background: 'var(--bg-panel)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '12px',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
    };

    return (
        <div className={`cyber-card ${className}`} style={cardStyle}>
            {/* Corner Accents */}
            <div style={{ position: 'absolute', top: 0, left: 0, width: '16px', height: '16px', borderTop: `2px solid ${getBorderColor()}`, borderLeft: `2px solid ${getBorderColor()}`, borderTopLeftRadius: '12px', opacity: 0.6 }} />
            <div style={{ position: 'absolute', bottom: 0, right: 0, width: '16px', height: '16px', borderBottom: `2px solid ${getBorderColor()}`, borderRight: `2px solid ${getBorderColor()}`, borderBottomRightRadius: '12px', opacity: 0.6 }} />

            {title && (
                <div style={{
                    padding: '14px 18px',
                    borderBottom: '1px solid var(--border-subtle)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                }}>
                    {icon && <span style={{ color: getBorderColor() }}>{icon}</span>}
                    <h3 style={{ margin: 0, fontSize: '14px', color: 'white', letterSpacing: '0.08em', fontFamily: 'var(--font-display)' }}>{title}</h3>
                </div>
            )}

            <div style={{ padding: '16px', flex: 1, overflowY: 'auto' }}>
                {children}
            </div>
        </div>
    );
};

export default CyberCard;
