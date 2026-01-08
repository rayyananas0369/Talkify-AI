import { useState } from "react";
import ModeToggle from "./components/ModeToggle";
import CameraFeed from "./components/CameraFeed";
import OutputBox from "./components/OutputBox";
import Features from "./components/Features";
import Footer from "./components/Footer";


function App() {
  const [mode, setMode] = useState("lip"); // default mode
  const [output, setOutput] = useState("");
  const [showGuides, setShowGuides] = useState(true);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div style={{ width: '80px' }}></div> {/* Spacer */}
        <div className="header-brand" style={{ textAlign: 'center' }}>
          <h1>Talkify AI</h1>
          <p>Multimodal AI for hearing and speech impaired users</p>
        </div>
        <nav>
          <a href="#about" className="btn-about" style={{ textDecoration: 'none' }}>
            About
          </a>
        </nav>
      </header>

      <main className="main-content">

        {/* Mode Toggle */}
        <ModeToggle onModeChange={setMode} />

        {/* Dashboard Grid */}
        <div className="dashboard-grid">
          {/* Camera Feed */}
          <CameraFeed mode={mode} setOutputText={setOutput} showGuides={showGuides} />

          {/* Output Box */}
          <OutputBox
            text={output}
            setOutputText={setOutput}
            showGuides={showGuides}
            setShowGuides={setShowGuides}
          />
        </div>

        {/* Features Section */}
        <section id="about" style={{ marginTop: '3rem' }}>
          <Features />
        </section>


      </main>
      <Footer />
    </div>
  );
}

export default App;
