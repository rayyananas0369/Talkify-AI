import React from 'react';

const ModeSelector = ({ currentMode, setMode }) => {
  const handleToggle = (selectedMode) => {
    setMode(selectedMode);
  };

  return (
    <div className="mode-selector-container" style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2rem' }}>
      <button
        onClick={() => handleToggle("lip")}
        className={`btn-mode ${currentMode === "lip" ? "active" : ""}`}
        style={{ padding: '0.75rem 1.5rem', borderRadius: '30px', border: 'none', background: currentMode === 'lip' ? '#4f46e5' : '#e5e7eb', color: currentMode === 'lip' ? 'white' : '#4b5563', cursor: 'pointer', fontWeight: '600' }}
      >
        Lip Reading
      </button>
      <button
        onClick={() => handleToggle("sign")}
        className={`btn-mode ${currentMode === "sign" ? "active" : ""}`}
        style={{ padding: '0.75rem 1.5rem', borderRadius: '30px', border: 'none', background: currentMode === 'sign' ? '#4f46e5' : '#e5e7eb', color: currentMode === 'sign' ? 'white' : '#4b5563', cursor: 'pointer', fontWeight: '600' }}
      >
        Sign Language
      </button>
    </div>
  );
};

export default ModeSelector;
