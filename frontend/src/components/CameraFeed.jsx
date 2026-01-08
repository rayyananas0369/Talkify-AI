import { useRef, useState, useEffect } from "react";
import { predictSign, predictLip } from "../services/api";

export default function CameraFeed({ mode, setOutputText, showGuides }) {
    const videoRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [showPopup, setShowPopup] = useState(false);
    const [isActive, setIsActive] = useState(false);
    const canvasOverlayRef = useRef(null);

    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ];

    const drawGuides = (landmarks, hand_rect) => {
        if (!canvasOverlayRef.current) return;
        const ctx = canvasOverlayRef.current.getContext("2d");
        const { width, height } = canvasOverlayRef.current;
        ctx.clearRect(0, 0, width, height);

        if (!landmarks && !hand_rect) return;

        // Draw YOLOv8 Hand Box
        if (hand_rect) {
            ctx.strokeStyle = "#f59e0b"; // Amber for box
            ctx.lineWidth = 2;
            const [x1, y1, x2, y2] = hand_rect;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = "#f59e0b";
            ctx.fillText("HAND", x1, y1 > 10 ? y1 - 5 : 10);
        }

        if (landmarks) {
            ctx.strokeStyle = "#10b981"; // Emerald for landmarks
            ctx.lineWidth = 2;

            if (mode === "sign" && landmarks.length === 21) {
                // Draw Hand Connections
                ctx.beginPath();
                HAND_CONNECTIONS.forEach(([start, end]) => {
                    const startPt = landmarks[start];
                    const endPt = landmarks[end];
                    if (startPt && endPt) {
                        ctx.moveTo(startPt.x * width, startPt.y * height);
                        ctx.lineTo(endPt.x * width, endPt.y * height);
                    }
                });
                ctx.stroke();
            }

            // Draw Dots
            ctx.fillStyle = "white";
            landmarks.forEach(lm => {
                const x = lm.x * width;
                const y = lm.y * height;
                ctx.beginPath();
                ctx.arc(x, y, mode === "lip" ? 2 : 3, 0, 2 * Math.PI);
                ctx.fill();
            });
        }
    };

    const handleStartClick = () => setShowPopup(true);

    const handleAllow = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) videoRef.current.srcObject = mediaStream;
            setStream(mediaStream);
            setIsActive(true);

            const interval = setInterval(async () => {
                if (!videoRef.current) return;

                const captureCanvas = document.createElement("canvas");
                captureCanvas.width = videoRef.current.videoWidth;
                captureCanvas.height = videoRef.current.videoHeight;
                captureCanvas.getContext("2d").drawImage(videoRef.current, 0, 0);

                if (canvasOverlayRef.current) {
                    canvasOverlayRef.current.width = captureCanvas.width;
                    canvasOverlayRef.current.height = captureCanvas.height;
                }

                captureCanvas.toBlob(async (blob) => {
                    if (!blob) return;
                    const formData = new FormData();
                    formData.append("file", blob, "frame.jpg");

                    try {
                        const data = mode === "lip"
                            ? await predictLip(formData)
                            : await predictSign(formData);

                        if (showGuides) {
                            drawGuides(data.landmarks, data.hand_rect);
                        } else if (canvasOverlayRef.current) {
                            const ctx = canvasOverlayRef.current.getContext("2d");
                            ctx.clearRect(0, 0, canvasOverlayRef.current.width, canvasOverlayRef.current.height);
                        }

                        if (data.confidence && data.confidence > 0.40) {
                            const charToShow = data.text === "_" ? " " : data.text;
                            setOutputText(charToShow);
                        }
                    } catch (err) {
                        console.error("Backend error:", err);
                    }
                }, 'image/jpeg', 0.8);
            }, 100);

            videoRef.current.intervalId = interval;
        } catch (err) {
            console.error("Camera access denied", err);
            alert("Camera access denied");
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
        setIsActive(false);
    };

    useEffect(() => {
        return () => stopCamera();
    }, []);

    return (
        <div className="card">
            <h2 className="card-title">Camera</h2>
            <div className="camera-container">
                <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            transform: 'scaleX(-1)',
                            display: isActive ? 'block' : 'none'
                        }}
                    />
                    {isActive && (
                        <canvas
                            ref={canvasOverlayRef}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                pointerEvents: 'none',
                                transform: 'scaleX(-1)'
                            }}
                        />
                    )}
                </div>
                {!isActive && (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#888' }}>
                        <p>Camera Stopped</p>
                    </div>
                )}
            </div>
            <div className="camera-controls">
                <button onClick={handleStartClick} className="btn btn-start" disabled={isActive}>Start</button>
                <button onClick={stopCamera} className="btn btn-stop" disabled={!isActive}>Stop</button>
            </div>

            {showPopup && (
                <div style={{
                    position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
                    background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
                }}>
                    <div className="card" style={{ maxWidth: '400px' }}>
                        <h2 className="text-xl font-bold mb-4" style={{ color: 'var(--primary)' }}>Security Check</h2>
                        <p className="mb-6">Do you want to allow camera access?</p>
                        <div className="flex gap-4 justify-center">
                            <button onClick={handleAllow} className="btn btn-start">Allow</button>
                            <button onClick={handleDeny} className="btn btn-stop">Deny</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
