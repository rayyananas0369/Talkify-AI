import { useRef, useState, useEffect } from "react";
import { Camera, StopCircle, RefreshCw, ShieldCheck, VideoOff, Eye, EyeOff } from "lucide-react";

export default function CameraFeed({ mode, setOutputText }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [showPopup, setShowPopup] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [showGuides, setShowGuides] = useState(true);

    const handleStartClick = () => {
        setShowPopup(true);
        setOutputText("Waiting for recognition...");
    };

    const handleAllow = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
            setStream(mediaStream);
            setIsProcessing(true);

            const interval = setInterval(async () => {
                if (!videoRef.current) return;

                const canvas = document.createElement("canvas");
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(videoRef.current, 0, 0);

                const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");

                const url =
                    mode === "lip"
                        ? "http://localhost:8000/predict/lip"
                        : "http://localhost:8000/predict/sign";

                try {
                    const res = await fetch(url, { method: "POST", body: formData });
                    const data = await res.json();
                    setOutputText(data.text);

                    // Visual Guides / Tracking Overlay
                    if (canvasRef.current && videoRef.current) {
                        const overlayCtx = canvasRef.current.getContext('2d');
                        // Match dimensions
                        canvasRef.current.width = videoRef.current.videoWidth;
                        canvasRef.current.height = videoRef.current.videoHeight;

                        overlayCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                        if (showGuides && data.hand_rect) {
                            const [x1, y1, x2, y2] = data.hand_rect;

                            // Draw Bounding Box
                            overlayCtx.strokeStyle = "#4f46e5"; // Indigo-600
                            overlayCtx.lineWidth = 4;
                            overlayCtx.beginPath();
                            overlayCtx.rect(x1, y1, x2 - x1, y2 - y1);
                            overlayCtx.stroke();

                            // Draw Label
                            overlayCtx.fillStyle = "#4f46e5";
                            overlayCtx.fillRect(x1, y1 - 30, x2 - x1, 30);
                            overlayCtx.fillStyle = "#ffffff";
                            overlayCtx.font = "bold 16px sans-serif";
                            overlayCtx.fillText("Tracking", x1 + 8, y1 - 8);
                        }
                    }

                } catch (err) {
                    console.error("Backend error:", err);
                }
            }, 200);

            videoRef.current.intervalId = interval;
        } catch (err) {
            console.error("Camera access denied", err);
        }

        setShowPopup(false);
    };

    const handleDeny = () => {
        setShowPopup(false);
        stopCamera();
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
        if (videoRef.current?.intervalId) {
            clearInterval(videoRef.current.intervalId);
        }
        setIsProcessing(false);
        // Clear canvas
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    return (
        <div className="card h-full flex flex-col overflow-hidden relative group">
            {/* Header */}
            <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-white">
                <div className="flex items-center gap-2">
                    <div className={`w-2.5 h-2.5 rounded-full ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-slate-300'}`} />
                    <h2 className="font-semibold text-slate-800">Camera Feed</h2>
                </div>

                <div className="flex items-center gap-3">
                    {/* Visual Guides Toggle */}
                    <button
                        onClick={() => setShowGuides(!showGuides)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${showGuides
                            ? "bg-indigo-100 text-indigo-700"
                            : "bg-slate-100 text-slate-500"
                            }`}
                        title={showGuides ? "Hide Visual Guides" : "Show Visual Guides"}
                    >
                        {showGuides ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                        {showGuides ? "Guides On" : "Guides Off"}
                    </button>

                    {isProcessing && (
                        <span className="text-xs font-medium text-indigo-600 bg-indigo-50 px-2 py-1 rounded inline-flex items-center gap-1">
                            <RefreshCw className="w-3 h-3 animate-spin" />
                            Processing
                        </span>
                    )}
                </div>
            </div>

            {/* Video Area */}
            <div className="relative bg-slate-900 w-full h-80 md:h-[480px] flex items-center justify-center overflow-hidden">
                {!stream && (
                    <div className="text-center p-6">
                        <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-500">
                            <VideoOff className="w-8 h-8" />
                        </div>
                        <p className="text-slate-400 text-sm">Camera is currently off</p>
                    </div>
                )}
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 rounded-lg ${!stream ? 'hidden' : ''}`}
                />
                <canvas
                    ref={canvasRef}
                    className={`absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 ${!stream ? 'hidden' : ''}`}
                />
            </div>

            {/* Controls */}
            <div className="p-4 bg-white flex justify-center gap-6">
                {!stream ? (
                    <button
                        onClick={handleStartClick}
                        className="px-8 py-2.5 bg-green-500 text-white rounded-lg font-bold shadow-md hover:bg-green-600 hover:scale-105 transition-all text-lg tracking-wide flex items-center gap-2"
                    >
                        Start
                    </button>
                ) : (
                    <button
                        onClick={stopCamera}
                        className="px-8 py-2.5 bg-red-500 text-white rounded-lg font-bold shadow-md hover:bg-red-600 hover:scale-105 transition-all text-lg tracking-wide flex items-center gap-2"
                    >
                        Stop
                    </button>
                )}
            </div>

            {/* Permission Popup */}
            {showPopup && (
                <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-white p-6 rounded-2xl shadow-2xl max-w-sm w-full animate-float">
                        <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4 text-indigo-600">
                            <ShieldCheck className="w-6 h-6" />
                        </div>
                        <h3 className="text-lg font-bold text-slate-800 text-center mb-2">Camera Permission</h3>
                        <p className="text-slate-500 text-center mb-6 text-sm">
                            We need access to your camera to analyze {mode === 'lip' ? 'lip movements' : 'signs'}. no video is stored.
                        </p>
                        <div className="grid grid-cols-2 gap-3">
                            <button
                                onClick={handleDeny}
                                className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 font-medium transition-colors"
                            >
                                Deny
                            </button>
                            <button
                                onClick={handleAllow}
                                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium transition-colors"
                            >
                                Allow Access
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
