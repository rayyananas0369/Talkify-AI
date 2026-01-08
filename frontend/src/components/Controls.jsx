import React from 'react';
import { Hand, Accessibility, Repeat, Smile } from "lucide-react";

const Controls = () => {
  const features = [
    {
      title: "Sign Language Recognition",
      desc: "Real-time translation of sign language gestures into text using advanced AI models.",
      icon: <Hand size={32} color="#f59e0b" />,
    },
    {
      title: "Lip Reading Technology",
      desc: "Cutting-edge lip reading algorithms convert visual speech into written text.",
      icon: <Smile size={32} color="#3b82f6" />,
    },
    {
      title: "Accessibility Focused",
      desc: "Designed specifically for hearing and speech impaired users with intuitive controls.",
      icon: <Accessibility size={32} color="#3b82f6" />,
    },
    {
      title: "Multi-mode Operation",
      desc: "Toggle between different input modes based on your needs and preferences.",
      icon: <Repeat size={32} color="#22c55e" />,
    },
  ];

  return (
    <div className="features-grid">
      {features.map((f, idx) => (
        <div key={idx} className="feature-card">
          <div className="feature-icon" style={{ display: 'flex', justifyContent: 'center' }}>
            {f.icon}
          </div>
          <h3>{f.title}</h3>
          <p>{f.desc}</p>
        </div>
      ))}
    </div>
  );
};

export default Controls;
