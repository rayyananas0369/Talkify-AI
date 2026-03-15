import { useState } from "react";
import ModeToggle from "./components/ModeToggle";
import CameraFeed from "./components/CameraFeed";
import OutputBox from "./components/OutputBox";
import Features from "./components/Features";

function App() {
  const [mode, setMode] = useState("sign");
  const [output, setOutput] = useState(""); // Reverted to single text
  const [draft, setDraft] = useState(""); // Real-time intermediate results
  const [status, setStatus] = useState("System Ready");
  const [fusionData, setFusionData] = useState(null);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-['Outfit']">
      <header className="bg-indigo-600 text-white shadow-lg py-4 px-6 relative">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between relative">
          <div className="hidden md:block w-20"></div>
          <div className="text-center flex-1">
            <h1 className="text-3xl font-bold tracking-tight mb-1">Talkify AI</h1>
            <p className="text-indigo-100 text-sm md:text-base font-medium opacity-90">
              Multimodal AI for hearing and speech impaired users
            </p>
          </div>
          <div className="mt-4 md:mt-0 w-20 flex justify-end">
            <a
              href="#about"
              className="bg-white text-indigo-600 hover:bg-indigo-50 font-semibold px-4 py-1.5 rounded-lg shadow-sm transition-all text-sm"
            >
              About
            </a>
          </div>
        </div>
      </header>

      <div className="flex justify-center mt-8 mb-4">
        <ModeToggle mode={mode} onModeChange={setMode} />
      </div>

      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 pb-12 grid lg:grid-cols-2 gap-8">
        <div className="flex flex-col gap-4">
          <CameraFeed
            mode={mode}
            setOutputText={setOutput}
            setDraftText={setDraft}
            setStatus={setStatus}
            status={status}
            setFusionData={setFusionData}
            fusionData={fusionData}
          />
        </div>
        <div className="flex flex-col gap-4">
          <OutputBox
            text={output}
            draft={draft}
            onClear={() => { setOutput(""); setDraft(""); }}
          />
        </div>
      </main>

      <section id="about" className="max-w-7xl mx-auto px-6 pb-12">
        <Features />
      </section>

      <footer className="bg-slate-900 text-slate-400 text-center py-6 mt-auto">
        <p className="text-sm">Talkify AI © 2025 | Empowering Communication</p>
      </footer>
    </div>
  );
}

export default App;
