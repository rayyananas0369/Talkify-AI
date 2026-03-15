import React from 'react';

const ModeSelector = ({ currentMode, setMode }) => {
  const handleToggle = (selectedMode) => {
    setMode(selectedMode);
  };

  return (
    <div className="mode-selector-container" style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2rem' }}>
      <button
        onClick={() => handleToggle("sign")}
        className={`btn-mode ${currentMode === "sign" ? "active" : ""}`}
        style={{ padding: '0.75rem 1.5rem', borderRadius: '30px', border: 'none', background: currentMode === 'sign' ? '#4f46e5' : '#e5e7eb', color: currentMode === 'sign' ? 'white' : '#4b5563', cursor: 'pointer', fontWeight: '600' }}
      >
        Sign Language
      </button>
      <button
        onClick={() => handleToggle("voice")}
        className={`btn-mode ${currentMode === "voice" ? "active" : ""}`}
        style={{ padding: '0.75rem 1.5rem', borderRadius: '30px', border: 'none', background: currentMode === 'voice' ? '#4f46e5' : '#e5e7eb', color: currentMode === 'voice' ? 'white' : '#4b5563', cursor: 'pointer', fontWeight: '600' }}
      >
        Voice Recognition
      </button>
    </div>
  );
};

export default ModeSelector;
