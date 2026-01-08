export default function OutputBox({ text, setOutputText, showGuides, setShowGuides }) {
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    alert("Text copied to clipboard!");
  };

  const handleClear = () => {
    setOutputText("");
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h2 className="card-title" style={{ margin: 0 }}>Output</h2>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
          <input
            type="checkbox"
            checked={showGuides}
            onChange={(e) => setShowGuides(e.target.checked)}
          />
          Show Visual Guides
        </label>
      </div>

      <textarea
        className="output-area"
        value={text}
        readOnly
        placeholder="Waiting for recognition..."
      />
      <div style={{ display: 'flex', gap: '1rem' }}>
        <button onClick={handleCopy} className="btn btn-start" style={{ flex: 1 }}>Copy Text</button>
        <button onClick={handleClear} className="btn btn-stop" style={{ flex: 1 }}>Clear Text</button>
      </div>
    </div>
  );
}
