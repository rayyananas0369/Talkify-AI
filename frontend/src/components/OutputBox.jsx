import { Copy, Trash2, Check, MessageSquare } from "lucide-react";
import { useState } from "react";

export default function OutputBox({ text, draft, onClear }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        if (!text) return;
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="card h-full flex flex-col relative overflow-hidden bg-white shadow-xl rounded-3xl border border-slate-100/50">
            {/* Header */}
            <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-white/80 backdrop-blur-md sticky top-0 z-10">
                <div className="flex items-center gap-3">
                    <div className="bg-indigo-600 p-2 rounded-xl text-white shadow-lg shadow-indigo-200">
                        <MessageSquare className="w-4 h-4" />
                    </div>
                    <div>
                        <h2 className="font-bold text-slate-800 leading-none">Transcription</h2>
                        <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-1 inline-block">Live Output</span>
                    </div>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-grow p-8 bg-slate-50/50 relative overflow-y-auto">
                {!text && !draft ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center">
                        <MessageSquare className="w-16 h-16 opacity-10 mb-4" />
                        <p className="text-sm font-medium">Your text will appear here as you speak.</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <p className="text-slate-700 text-xl md:text-2xl font-medium leading-relaxed whitespace-pre-wrap">
                            {text}
                            {draft && (
                                <span className="text-indigo-400 italic opacity-60 ml-2 animate-pulse">
                                    {draft}...
                                </span>
                            )}
                        </p>
                    </div>
                )}
            </div>

            {/* Actions */}
            <div className="p-4 bg-white border-t border-slate-100 flex gap-3">
                <button
                    onClick={handleCopy}
                    disabled={!text}
                    className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl font-bold transition-all ${!text
                        ? "bg-slate-50 text-slate-300 cursor-not-allowed border border-slate-100"
                        : "bg-slate-900 text-white hover:bg-slate-800 active:scale-95 shadow-lg shadow-slate-200"
                        }`}
                >
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                    {copied ? "Copied" : "Copy Output"}
                </button>
                <button
                    disabled={!text && !draft}
                    onClick={onClear}
                    className={`px-4 py-3 rounded-2xl font-bold transition-all border ${!text && !draft
                        ? "bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed"
                        : "bg-white text-red-500 border-red-100 hover:bg-red-50 active:scale-95"
                        }`}
                >
                    <Trash2 className="w-5 h-5" />
                </button>
            </div>
        </div>
    );
}
