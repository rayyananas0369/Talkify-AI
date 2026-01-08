import { useState } from "react";

export default function ModeToggle({ onModeChange }) {
  const [mode, setMode] = useState("lip");

  const handleToggle = (selectedMode) => {
    setMode(selectedMode);
    onModeChange(selectedMode);
  };

  return (
    <div className="mode-toggle">
      <button
        onClick={() => handleToggle("lip")}
        className={`btn-mode ${mode === "lip" ? "active" : ""}`}
      >
        Lip Reading
      </button>
      <button
        onClick={() => handleToggle("sign")}
        className={`btn-mode ${mode === "sign" ? "active" : ""}`}
      >
        Sign Language
      </button>
    </div>
  );
}
