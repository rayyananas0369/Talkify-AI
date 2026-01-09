import { Video, Hand } from "lucide-react";

export default function ModeToggle({ mode, onModeChange }) {
    return (
        <div className="bg-slate-100 p-1.5 rounded-xl inline-flex items-center shadow-inner border border-slate-200">
            <button
                onClick={() => onModeChange("lip")}
                className={`px-5 py-2.5 rounded-lg flex items-center gap-2 text-sm font-semibold transition-all duration-200 ${mode === "lip"
                        ? "bg-white text-indigo-600 shadow-sm ring-1 ring-black/5"
                        : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                    }`}
            >
                <Video className="w-4 h-4" />
                Lip Reading
            </button>
            <button
                onClick={() => onModeChange("sign")}
                className={`px-5 py-2.5 rounded-lg flex items-center gap-2 text-sm font-semibold transition-all duration-200 ${mode === "sign"
                        ? "bg-white text-indigo-600 shadow-sm ring-1 ring-black/5"
                        : "text-slate-500 hover:text-slate-700 hover:bg-slate-200/50"
                    }`}
            >
                <Hand className="w-4 h-4" />
                Sign Language
            </button>
        </div>
    );
}
