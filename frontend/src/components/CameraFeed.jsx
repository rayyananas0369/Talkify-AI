import { useRef, useState, useEffect } from "react";
import { Camera, VideoOff, Eye, EyeOff, ShieldCheck, RefreshCw, Mic, MicOff } from "lucide-react";

export default function CameraFeed({ mode, setOutputText, setDraftText, status, setStatus, setFusionData, fusionData }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [showPopup, setShowPopup] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [showGuides, setShowGuides] = useState(true);
    const [cameraError, setCameraError] = useState(null);
    const [noiseLevel, setNoiseLevel] = useState(0);
    const [liveGuess, setLiveGuess] = useState("");

    // Audio context for Voice Recognition
    const audioContextRef = useRef(null);
    const pcmBufferRef = useRef([]);

    // Sign Language specific refs
    const lastDetectedChar = useRef("");
    const framesHeld = useRef(0);

    const handleStartClick = () => {
        setCameraError(null);
        setShowPopup(true);
        if (typeof setCurrentText === 'function') {
            setCurrentText("Waiting for recognition...");
        }
    };

    const handleAllow = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: mode === 'voice'
            });
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
            setStream(mediaStream);
            setIsProcessing(true);
            setCameraError(null);

            if (mode === 'voice') {
                initAudioCapture(mediaStream);
            }
        } catch (err) {
            console.error("Device access denied", err);
            setCameraError(err.name === "NotAllowedError" ? "Camera access was denied." : "Could not access device: " + err.message);
        }
        setShowPopup(false);
    };

    const initAudioCapture = (mediaStream) => {
        try {
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContextRef.current.createMediaStreamSource(mediaStream);
            const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);

            pcmBufferRef.current = [];
            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                pcmBufferRef.current.push(new Float32Array(inputData));
            };

            source.connect(processor);
            processor.connect(audioContextRef.current.destination);
        } catch (e) {
            console.error("Audio init error:", e);
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        setIsProcessing(false);
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        setLiveGuess("");
        if (typeof setCurrentText === 'function') setCurrentText("");
    };

    useEffect(() => {
        let isMounted = true;
        let animationFrameId;

        const FPS = 25;
        const frameInterval = 1000 / FPS;
        let lastFrameTime = 0;

        const processFrame = async (timestamp) => {
            if (!isMounted || !isProcessing || !videoRef.current || !stream || videoRef.current.videoWidth === 0) {
                if (isMounted && isProcessing) {
                    animationFrameId = requestAnimationFrame(processFrame);
                }
                return;
            }

            if (timestamp - lastFrameTime < frameInterval) {
                animationFrameId = requestAnimationFrame(processFrame);
                return;
            }
            lastFrameTime = timestamp;

            const canvas = document.createElement("canvas");
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(videoRef.current, 0, 0);

            try {
                const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");

                if (mode === 'voice' && pcmBufferRef.current.length > 0) {
                    const chunksToProcess = [...pcmBufferRef.current];
                    pcmBufferRef.current = [];

                    const totalLength = chunksToProcess.reduce((acc, b) => acc + b.length, 0);
                    const combined = new Float32Array(totalLength);
                    let offset = 0;
                    for (const b of chunksToProcess) {
                        combined.set(b, offset);
                        offset += b.length;
                    }
                    formData.append("audio", new Blob([combined.buffer], { type: "application/octet-stream" }));
                }

                const url = mode === 'voice' ? "http://127.0.0.1:8005/predict/voice" : "http://127.0.0.1:8005/predict/sign";
                const res = await fetch(url, { method: "POST", body: formData });

                if (!isMounted) return;
                const data = await res.json();

                if (data.status && typeof setStatus === 'function') {
                    setStatus(data.status);
                }

                if (data.fusion_status) {
                    if (setFusionData) setFusionData(data.fusion_status);
                    if (data.fusion_status.noise_level !== undefined) {
                        setNoiseLevel(data.fusion_status.noise_level);
                    }
                }

                if (data.text) {
                    if (mode === 'sign') {
                        const charToAdd = data.text === "_" ? " " : data.text;
                        if (charToAdd !== lastDetectedChar.current) {
                            setOutputText(prev => (prev === "Waiting for recognition..." ? "" : prev) + charToAdd);
                            lastDetectedChar.current = charToAdd;
                            framesHeld.current = 1;
                        } else {
                            framesHeld.current++;
                            if (framesHeld.current >= 25) {
                                setOutputText(prev => prev + charToAdd);
                                framesHeld.current = 1;
                            }
                        }
                    } else {
                        // Voice Recognition Mode Processing
                        if (data.is_final && data.text) {
                            // Append phrases clearly separated by a newline
                            setOutputText(prev => {
                                if (prev === "Waiting for recognition..." || prev === "Recognition complete") return data.text;
                                const current = prev.trim();
                                if (current === "") return data.text;
                                return `${current}\n${data.text}`;
                            });
                        } else {
                            // Drafting
                            if (typeof setDraftText === 'function') setDraftText(data.text);
                        }
                    }
                } else if (data.status && (data.status.includes("Finding Face") || data.status.includes("Waiting"))) {
                    lastDetectedChar.current = "";
                    framesHeld.current = 0;
                    if (mode === 'voice') {
                        if (typeof setDraftText === 'function') setDraftText("");
                        setNoiseLevel(0);
                    }
                }

                if (canvasRef.current && videoRef.current && showGuides) {
                    const overlayCtx = canvasRef.current.getContext('2d');
                    overlayCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                    if (data.landmarks && data.landmarks.length > 0) {
                        const width = canvasRef.current.width;
                        const height = canvasRef.current.height;

                        // 1. Draw "Sleek Voice Contour" (Dynamic Colors)
                        overlayCtx.globalAlpha = 0.8;

                        // Status-based coloring
                        let statusColor = "#6366f1"; // Default Indigo
                        if (data.status === "READY") statusColor = "#10b981";    // Emerald Green
                        if (data.status === "LISTENING") statusColor = "#3b82f6"; // Bright Blue
                        if (data.status === "Processing...") statusColor = "#94a3b8"; // Slate Gray

                        if (mode === 'sign') {
                            overlayCtx.strokeStyle = statusColor;
                            const palmConnections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
                            palmConnections.forEach(([i, j]) => {
                                const lm1 = data.landmarks[i];
                                const lm2 = data.landmarks[j];
                                if (lm1 && lm2) {
                                    overlayCtx.beginPath();
                                    overlayCtx.moveTo(lm1.x * width, lm1.y * height);
                                    overlayCtx.lineTo(lm2.x * width, lm2.y * height);
                                    overlayCtx.stroke();
                                }
                            });
                        } else if (mode === 'voice') {
                            // Draw outer voice loop
                            overlayCtx.beginPath();
                            overlayCtx.lineWidth = 2.0;
                            overlayCtx.strokeStyle = statusColor;
                            overlayCtx.moveTo(data.landmarks[0].x * width, data.landmarks[0].y * height);
                            for (let i = 1; i <= 11; i++) {
                                overlayCtx.lineTo(data.landmarks[i].x * width, data.landmarks[i].y * height);
                            }
                            overlayCtx.closePath();
                            overlayCtx.stroke();

                            // Draw inner voice loop
                            overlayCtx.beginPath();
                            overlayCtx.lineWidth = 1.0;
                            overlayCtx.strokeStyle = statusColor;
                            overlayCtx.setLineDash([2, 2]); // Dotted inner for professional look
                            overlayCtx.moveTo(data.landmarks[12].x * width, data.landmarks[12].y * height);
                            for (let i = 13; i < data.landmarks.length; i++) {
                                overlayCtx.lineTo(data.landmarks[i].x * width, data.landmarks[i].y * height);
                            }
                            overlayCtx.closePath();
                            overlayCtx.stroke();
                            overlayCtx.setLineDash([]); // Reset dash
                        }

                        overlayCtx.globalAlpha = 1.0;

                        // Reset shadow for next frame
                        overlayCtx.shadowBlur = 0;
                    }
                }
            } catch (err) {
                console.error("Frame processing error:", err);
            }

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
        };
    }, [stream, isProcessing, mode, showGuides, setOutputText, setDraftText, setStatus]);

    return (
        <div className="card h-full flex flex-col overflow-hidden relative group">
            <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-white">
                <div className="flex items-center gap-2">
                    <div className={`w-2.5 h-2.5 rounded-full ${isProcessing ? 'bg-green-500 animate-pulse' : 'bg-slate-300'}`} />
                    <h2 className="font-semibold text-slate-800">Camera Feed</h2>
                </div>

                <div className="flex items-center gap-3">
                    {mode === 'voice' && fusionData && (
                        <div className="flex gap-2 text-[10px] font-black uppercase tracking-widest px-3 py-1 bg-slate-50 rounded-full border border-slate-100">
                            <span className="text-indigo-500">V: {(fusionData.visual_confidence * 100).toFixed(0)}%</span>
                            <span className="text-emerald-500">A: {(fusionData.audio_confidence * 100).toFixed(0)}%</span>
                        </div>
                    )}
                    <button
                        onClick={() => setShowGuides(!showGuides)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold transition-all ${showGuides ? "bg-indigo-100 text-indigo-700" : "bg-slate-100 text-slate-500"}`}
                    >
                        {showGuides ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                        {showGuides ? "Guides On" : "Guides Off"}
                    </button>
                    {isProcessing && <RefreshCw className="w-4 h-4 animate-spin text-indigo-500" />}
                </div>
            </div>

            <div className="relative bg-slate-900 w-full h-80 md:h-[480px] flex items-center justify-center overflow-hidden">
                {!stream && (
                    <div className="text-center p-6 text-slate-400">
                        <VideoOff className="w-12 h-12 mx-auto mb-4 opacity-20" />
                        <p>Camera is currently off</p>
                    </div>
                )}
                <video ref={videoRef} autoPlay playsInline muted className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 ${!stream ? 'hidden' : ''}`} />
                <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100 ${!stream ? 'hidden' : ''}`} />


                {stream && status && (
                    <div className="absolute bottom-6 left-6 flex items-center gap-3 bg-slate-900/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/10 shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className={`w-3 h-3 rounded-full shadow-[0_0_12px_rgba(255,255,255,0.3)] ${status === "READY" ? "bg-emerald-500 shadow-emerald-500/50" :
                            status === "LISTENING" ? "bg-blue-500 animate-pulse shadow-blue-500/50" :
                                status === "Processing..." ? "bg-amber-500 animate-spin" : "bg-slate-400"
                            }`} />
                        <span className={`text-[11px] font-black uppercase tracking-[0.2em] ${status === "READY" ? "text-emerald-400" :
                            status === "LISTENING" ? "text-blue-400" :
                                status === "Processing..." ? "text-amber-400" : "text-slate-300"
                            }`}>
                            {status}
                        </span>
                    </div>
                )}
            </div>

            <div className="p-4 bg-white flex justify-center gap-4">
                {!stream ? (
                    <button onClick={handleStartClick} className="px-8 py-2 bg-indigo-600 text-white rounded-lg font-bold hover:bg-indigo-700 transition-all">Start Camera</button>
                ) : (
                    <button onClick={stopCamera} className="px-8 py-2 bg-red-500 text-white rounded-lg font-bold hover:bg-red-600 transition-all">Stop Camera</button>
                )}
            </div>

            {showPopup && (
                <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-white p-6 rounded-2xl shadow-xl max-w-sm w-full">
                        <h3 className="text-lg font-bold mb-4">Device Access</h3>
                        <p className="text-slate-500 text-sm mb-6">We need access to your {mode === 'voice' ? 'camera and microphone' : 'camera'} for recognition.</p>
                        <div className="flex gap-3">
                            <button onClick={() => setShowPopup(false)} className="flex-1 py-2 bg-slate-100 rounded-lg font-bold">Cancel</button>
                            <button onClick={handleAllow} className="flex-1 py-2 bg-indigo-600 text-white rounded-lg font-bold">Allow</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
