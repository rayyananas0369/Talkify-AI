import React from 'react';
import OverlayToggle from './OverlayToggle';

const PredictionBox = ({ output, setOutput, showGuides, setShowGuides }) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(output);
    alert('Output copied to clipboard!');
  };

  const handleClear = () => {
    setOutput('');
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 className="card-title" style={{ margin: 0 }}>Prediction</h2>
        <OverlayToggle showGuides={showGuides} setShowGuides={setShowGuides} />
      </div>

      <textarea
        className="output-area"
        value={output}
        readOnly
        placeholder="Waiting for recognition..."
        style={{ width: '100%', minHeight: '150px', marginBottom: '1rem', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-light)', resize: 'none' }}
      />

      <div style={{ display: 'flex', gap: '1rem' }}>
        <button onClick={handleCopy} className="btn btn-copy" style={{ flex: 1, backgroundColor: '#4f46e5', color: 'white' }}>Copy Text</button>
        <button onClick={handleClear} className="btn btn-clear" style={{ flex: 1, backgroundColor: '#ef4444', color: 'white' }}>Clear Text</button>
      </div>
    </div>
  );
};

export default PredictionBox;
