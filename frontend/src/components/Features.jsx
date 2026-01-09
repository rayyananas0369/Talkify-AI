import { Hand, Accessibility, Repeat, Languages } from "lucide-react";

export default function Features() {
    const features = [
        {
            title: "Sign Language Recognition",
            desc: "Real-time translation of sign language gestures into text using advanced AI models.",
            icon: <Hand className="w-6 h-6 text-indigo-600" />,
            bg: "bg-indigo-50"
        },
        {
            title: "Lip Reading Technology",
            desc: "Cutting-edge lip reading algorithms convert visual speech into written text.",
            icon: <Languages className="w-6 h-6 text-pink-600" />,
            bg: "bg-pink-50"
        },
        {
            title: "Accessibility Focused",
            desc: "Designed specifically for hearing and speech impaired users with intuitive controls.",
            icon: <Accessibility className="w-6 h-6 text-teal-600" />,
            bg: "bg-teal-50"
        },
        {
            title: "Multi-mode Operation",
            desc: "Toggle between different input modes based on your needs and preferences.",
            icon: <Repeat className="w-6 h-6 text-orange-600" />,
            bg: "bg-orange-50"
        },
    ];

    return (
        <div className="mt-8">
            <div className="text-center mb-10">
                <h2 className="text-2xl font-bold text-slate-900">Powered by Advanced AI</h2>
                <p className="text-slate-500 mt-2">Connecting people through seamless, real-time communication tools.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {features.map((f, idx) => (
                    <div
                        key={idx}
                        className="p-6 bg-white rounded-2xl shadow-sm border border-slate-100 hover:shadow-xl hover:scale-105 transition-all duration-300 ease-out flex flex-col items-center text-center group"
                    >
                        <div className={`w-14 h-14 ${f.bg} rounded-2xl flex items-center justify-center mb-4 group-hover:rotate-6 transition-transform`}>
                            {f.icon}
                        </div>
                        <h3 className="font-bold text-lg text-slate-800 mb-2">{f.title}</h3>
                        <p className="text-slate-500 text-sm leading-relaxed">{f.desc}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
