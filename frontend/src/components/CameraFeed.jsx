import { useRef, useState, useEffect } from "react";
import { Camera, StopCircle, RefreshCw, ShieldCheck, VideoOff, Eye, EyeOff } from "lucide-react";

export default function CameraFeed({ mode, setOutputText, status, setStatus }) {
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
        } catch (err) {
            console.error("Camera access denied", err);
        }
        setShowPopup(false);
    };

    const lastDetectedChar = useRef("");
    const framesHeld = useRef(0);

    useEffect(() => {
        let isMounted = true;
        let animationFrameId;

        const processFrame = async () => {
            if (!isMounted || !isProcessing) return;

            if (!videoRef.current || !stream || videoRef.current.videoWidth === 0) {
                // Keep the loop alive even if video isn't ready
                animationFrameId = requestAnimationFrame(processFrame);
                return;
            }

            const canvas = document.createElement("canvas");
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(videoRef.current, 0, 0);

            const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));
            const formData = new FormData();
            formData.append("file", blob, "frame.jpg");

            const currentMode = mode;
            const url =
                currentMode === "lip"
                    ? "http://localhost:8000/predict/lip"
                    : "http://localhost:8000/predict/sign";

            try {
                const res = await fetch(url, { method: "POST", body: formData });
                if (!isMounted) return;

                const data = await res.json();
                if (!isMounted) return;

                if (setStatus) setStatus(data.status);

                if (data.text) {
                    if (currentMode === 'lip') {
                        // Sentence-level replacement for Lip Reading
                        setOutputText(data.text);
                    } else {
                        // Character-level appending for Sign Language
                        const charToAdd = data.text === "_" ? " " : data.text;

                        if (charToAdd !== lastDetectedChar.current) {
                            setOutputText(prev => {
                                const baseText = (prev === "Waiting for recognition..." || prev === "") ? "" : prev;
                                return baseText + charToAdd;
                            });
                            lastDetectedChar.current = charToAdd;
                            framesHeld.current = 1;
                        } else {
                            framesHeld.current += 1;
                            if (framesHeld.current >= 25) {
                                setOutputText(prev => prev + charToAdd);
                                framesHeld.current = 1;
                            }
                        }
                    }
                } else if (data.status && (data.status.includes("Finding Hand") || data.status.includes("Finding Face"))) {
                    lastDetectedChar.current = "";
                    framesHeld.current = 0;
                }

                if (canvasRef.current && videoRef.current) {
                    const overlayCtx = canvasRef.current.getContext('2d');
                    canvasRef.current.width = videoRef.current.videoWidth;
                    canvasRef.current.height = videoRef.current.videoHeight;
                    overlayCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                    if (showGuides) {
                        const width = canvasRef.current.width;
                        const height = canvasRef.current.height;

                        if (currentMode === 'sign' && data.hand_rect) {
                            const [x1, y1, x2, y2] = data.hand_rect;
                            overlayCtx.strokeStyle = "#4f46e5";
                            overlayCtx.lineWidth = 3;
                            overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                            overlayCtx.fillStyle = "#4f46e5";
                            overlayCtx.fillRect(x1, y1 - 25, 80, 25);
                            overlayCtx.fillStyle = "#ffffff";
                            overlayCtx.font = "bold 14px sans-serif";
                            overlayCtx.fillText("HAND", x1 + 5, y1 - 8);
                        }

                        if (currentMode === 'sign' && data.landmarks && data.landmarks.length > 0) {
                            overlayCtx.fillStyle = "#10b981";
                            overlayCtx.strokeStyle = "#10b981";
                            overlayCtx.lineWidth = 2;
                            const connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
                            connections.forEach(([i, j]) => {
                                const lm1 = data.landmarks[i];
                                const lm2 = data.landmarks[j];
                                if (lm1 && lm2) {
                                    overlayCtx.beginPath();
                                    overlayCtx.moveTo(lm1.x * width, lm1.y * height);
                                    overlayCtx.lineTo(lm2.x * width, lm2.y * height);
                                    overlayCtx.stroke();
                                }
                            });
                            data.landmarks.forEach(lm => {
                                overlayCtx.beginPath();
                                overlayCtx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
                                overlayCtx.fill();
                            });
                        }

                        if (currentMode === 'lip' && data.landmarks && data.landmarks.length > 0) {
                            overlayCtx.fillStyle = "#ec4899";
                            data.landmarks.forEach(lm => {
                                overlayCtx.beginPath();
                                overlayCtx.arc(lm.x * width, lm.y * height, 2, 0, 2 * Math.PI);
                                overlayCtx.fill();
                            });
                        }
                    }
                }
            } catch (err) {
                console.error("Backend error:", err);
            }

            // Schedule NEXT frame only AFTER this one is done
            if (isMounted && isProcessing) {
                animationFrameId = requestAnimationFrame(processFrame);
            }
        };

        if (stream && isProcessing) {
            processFrame();
        }

        return () => {
            isMounted = false;
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
        };
    }, [stream, isProcessing, mode, showGuides, setOutputText, setStatus]);

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

                {/* Status Indicator Overlay */}
                {stream && (
                    <div className="absolute top-4 left-4 z-10">
                        <div className="bg-slate-900/80 backdrop-blur-md text-white px-3 py-1.5 rounded-full border border-slate-700/50 flex items-center gap-2 shadow-xl animate-in fade-in slide-in-from-left-2">
                            <div className={`w-2 h-2 rounded-full ${status?.includes('Ready') ? 'bg-green-500' : 'bg-amber-500 animate-pulse'}`} />
                            <span className="text-xs font-bold tracking-tight uppercase">{status || "Initializing..."}</span>
                        </div>
                    </div>
                )}
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
