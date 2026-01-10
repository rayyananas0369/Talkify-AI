import { useState } from "react";
import ModeToggle from "./components/ModeToggle";
import CameraFeed from "./components/CameraFeed";
import OutputBox from "./components/OutputBox";
import Features from "./components/Features";

function App() {
  const [mode, setMode] = useState("lip");
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState("System Ready");

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-['Outfit']">
      {/* Header - Matching the Concept Image */}
      <header className="bg-indigo-600 text-white shadow-lg py-4 px-6 relative">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between relative">

          {/* Placeholder for left balance if needed, or just center the title */}
          <div className="hidden md:block w-20"></div>

          {/* Centered Title */}
          <div className="text-center flex-1">
            <h1 className="text-3xl font-bold tracking-tight mb-1">Talkify AI</h1>
            <p className="text-indigo-100 text-sm md:text-base font-medium opacity-90">
              Multimodal AI for hearing and speech impaired users
            </p>
          </div>

          {/* Right: About Button */}
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

      {/* Mode Toggle Section */}
      <div className="flex justify-center mt-8 mb-4">
        <ModeToggle mode={mode} onModeChange={setMode} />
      </div>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 pb-12 grid lg:grid-cols-2 gap-8">

        {/* Camera Feed Column */}
        <div className="flex flex-col gap-4">
          <CameraFeed mode={mode} setOutputText={setOutput} setStatus={setStatus} status={status} />
        </div>

        {/* Output Column */}
        <div className="flex flex-col gap-4">
          <OutputBox text={output} onClear={() => setOutput("")} />
        </div>

      </main>

      {/* Features / Details Section */}
      <section id="about" className="max-w-7xl mx-auto px-6 pb-12">
        <Features />
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 text-center py-6 mt-auto">
        <p className="text-sm">Talkify AI © 2025 | Empowering Communication</p>
      </footer>
    </div>
  );
}

export default App;
