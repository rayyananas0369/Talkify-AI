import { Copy, Trash2, Check, MessageSquare } from "lucide-react";
import { useState } from "react";

export default function OutputBox({ text }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        if (!text) return;
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="card h-full flex flex-col relative overflow-hidden">
            {/* Header */}
            <div className="p-4 border-b border-slate-100 flex items-center gap-2 bg-white">
                <div className="bg-indigo-100 p-1.5 rounded-lg text-indigo-600">
                    <MessageSquare className="w-4 h-4" />
                </div>
                <h2 className="font-semibold text-slate-800">Transcription Output</h2>
            </div>

            {/* Text Area */}
            <div className="flex-grow p-4 bg-slate-50 relative">
                {text ? (
                    <textarea
                        readOnly
                        value={text}
                        className="w-full h-full bg-transparent border-none resize-none focus:ring-0 text-slate-700 text-lg leading-relaxed font-medium"
                    />
                ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 p-6 text-center select-none">
                        <MessageSquare className="w-12 h-12 mb-3 opacity-20" />
                        <p className="text-sm">Processed speech will appear here automatically.</p>
                    </div>
                )}
            </div>

            {/* Actions */}
            <div className="p-4 bg-white border-t border-slate-100 flex gap-4 justify-center md:justify-start">
                <button
                    onClick={handleCopy}
                    disabled={!text}
                    className={`flex items-center justify-center gap-2 px-6 py-2 rounded-lg font-bold shadow-md transition-all text-white ${!text
                            ? "bg-slate-300 cursor-not-allowed"
                            : "bg-green-500 hover:bg-green-600 hover:scale-105"
                        }`}
                >
                    {copied ? <Check className="w-4 h-4" /> : null}
                    {copied ? "Copied" : "Copy Text"}
                </button>
                <button
                    disabled={!text}
                    className={`flex items-center justify-center gap-2 px-6 py-2 rounded-lg font-bold shadow-md transition-all text-white ${!text
                            ? "bg-slate-300 cursor-not-allowed"
                            : "bg-red-500 hover:bg-red-600 hover:scale-105"
                        }`}
                    onClick={() => { /* In a real app we'd lift this state up or pass a clearer function */ }}
                >
                    Clear Text
                </button>
            </div>
        </div>
    );
}
