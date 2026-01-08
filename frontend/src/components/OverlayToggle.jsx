import React from 'react';

const OverlayToggle = ({ showGuides, setShowGuides }) => {
    return (
        <div className="overlay-toggle" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
            <input
                type="checkbox"
                id="toggle-guides"
                checked={showGuides}
                onChange={(e) => setShowGuides(e.target.checked)}
            />
            <label htmlFor="toggle-guides" style={{ cursor: 'pointer' }}>Show Visual Guides</label>
        </div>
    );
};

export default OverlayToggle;
