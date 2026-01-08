import React, { useState } from 'react';
import CameraFeed from './components/CameraFeed';
import PredictionBox from './components/PredictionBox';
import ModeSelector from './components/ModeSelector';
import Controls from './components/Controls';
import Footer from './components/Footer';
import './styles/main.css';

function App() {
  const [mode, setMode] = useState('sign'); // 'sign' or 'lip'
  const [output, setOutput] = useState('');
  const [showGuides, setShowGuides] = useState(true);

  return (
    <div className="app-container">
      <header className="main-header">
        <h1 className="header-title">Talkify AI</h1>
        <p className="header-subtitle">Empowering accessibility through real-time communication</p>
        <button className="about-btn">About</button>
      </header>

      <main className="main-content">
        <ModeSelector currentMode={mode} setMode={setMode} />

        <div className="dashboard-grid">
          <CameraFeed
            mode={mode}
            showGuides={showGuides}
            setShowGuides={setShowGuides} // Used for OverlayToggle
            onPrediction={(text) => setOutput(prev => prev + ' ' + text)}
          />
          <PredictionBox
            output={output}
            setOutput={setOutput}
            showGuides={showGuides}
            setShowGuides={setShowGuides}
          />
        </div>

        <Controls />
      </main>

      <Footer />
    </div>
  );
}

export default App;
