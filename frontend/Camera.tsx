import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Camera, CameraOff, LoaderCircle } from "lucide-react";
import { LandmarkOverlay, RecognitionSummary } from "./components";
import Scratchpad from "./Scratchpad";
import type { ModelId, Recognition } from "./types";

export default function CameraView() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<Recognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [aspectRatio, setAspectRatio] = useState(16 / 9);
  const [fps, setFps] = useState(0);
  const [model, setModel] = useState<ModelId>("knn");

  function stop() {
    session.current++;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setRunning(false);
    setStarting(false);
    setResult(null);
    setFps(0);
  }

  useEffect(() => {
    return () => {
      session.current++;
      stream.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("The camera is not available in this environment.");
      return;
    }
    const current = ++session.current;
    setStarting(true);
    setError(null);
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (current !== session.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      video.current!.srcObject = media;
      await video.current!.play();
      if (current !== session.current) return;
      setAspectRatio(video.current!.videoWidth / video.current!.videoHeight);
      setRunning(true);
    } catch (reason) {
      if (current !== session.current) return;
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = null;
      const name = (reason as DOMException).name;
      setError(
        name === "NotAllowedError"
          ? "Camera access was denied. Allow access in your system settings and try again."
          : name === "NotFoundError"
            ? "No camera found. Connect a camera and try again."
            : `Could not start the camera: ${(reason as Error).message}`,
      );
    } finally {
      if (current === session.current) setStarting(false);
    }
  }

  useEffect(() => {
    if (!running || !window.desktop) return;
    const element = video.current!;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d")!;
    let canceled = false;
    // Keep a second frame queued so the next JPEG is encoded and sent while the
    // worker is still processing the previous one. With a single frame in
    // flight, a round trip slightly over one camera frame skips every other frame.
    const maxInFlight = 2;
    let inFlight = 0;
    let callback = 0;
    let completedFrames = 0;
    let sampleStarted = 0;

    async function analyzeFrame() {
      if (inFlight >= maxInFlight || !element.videoWidth) return;
      inFlight++;
      try {
        const width = Math.min(640, element.videoWidth);
        canvas.width = width;
        canvas.height = Math.round(element.videoHeight * (width / element.videoWidth));
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        const recognition = await window.desktop!.recognizeFrame(
          canvas.toDataURL("image/jpeg", 0.85),
          model,
        );
        if (!canceled) {
          setResult(recognition);
          setError(null);
          const now = performance.now();
          if (!sampleStarted) sampleStarted = now;
          else completedFrames++;
          if (now - sampleStarted >= 1000) {
            setFps(Math.round((completedFrames * 1000) / (now - sampleStarted)));
            completedFrames = 0;
            sampleStarted = now;
          }
        }
      } catch (reason) {
        if (!canceled) {
          setError((reason as Error).message);
          canceled = true;
          element.cancelVideoFrameCallback(callback);
        }
      } finally {
        inFlight--;
      }
    }

    function schedule() {
      callback = element.requestVideoFrameCallback(() => {
        if (canceled) return;
        schedule();
        void analyzeFrame();
      });
    }
    schedule();
    return () => {
      canceled = true;
      element.cancelVideoFrameCallback(callback);
    };
    // Restarting on a model change also discards results still in flight from the old model.
  }, [running, model]);

  return (
    <div className="view">
      <div className="view-heading">
        <div>
          <h1>Live camera</h1>
          <p>Show a sign to the camera and your hand is analyzed continuously.</p>
        </div>
        <span className={`status-pill ${running ? "live" : ""}`}>
          {running ? "Camera on" : "Camera off"}
        </span>
      </div>
      <div className="content-grid">
        <section className="preview-card" aria-label="Camera feed">
          <div className="camera-stage">
            <div className="camera-frame" style={{ "--ratio": aspectRatio } as CSSProperties}>
              <video
                ref={video}
                muted
                playsInline
                className="camera-video"
                aria-label="Live camera feed"
              />
              {running && <LandmarkOverlay result={result} mirrored />}
            </div>
            {!running && (
              <div className="empty-state">
                <CameraOff size={34} strokeWidth={1.4} />
                <strong>The camera is off</strong>
                <span>Start the camera to see your signs here.</span>
              </div>
            )}
          </div>
          <div className="preview-footer">
            <span>
              {running ? "Hand analysis runs locally" : "No video is recorded or stored"}
              {running && window.desktop && (
                <>
                  {" "}
                  · <span aria-label="Tracking frame rate">{fps} FPS</span>
                </>
              )}
            </span>
            <div className="preview-controls">
              <label className="model-select">
                Model
                <select
                  value={model}
                  onChange={(event) => {
                    setModel(event.target.value as ModelId);
                    setResult(null);
                  }}
                >
                  <option value="knn">KNN</option>
                  <option value="cnn">CNN</option>
                </select>
              </label>
              <button
                className="primary-button"
                type="button"
                disabled={starting}
                onClick={() => (running ? stop() : void start())}
              >
                {starting ? (
                  <LoaderCircle className="spin" size={17} />
                ) : running ? (
                  <CameraOff size={17} />
                ) : (
                  <Camera size={17} />
                )}
                {starting ? "Starting…" : running ? "Stop camera" : "Start camera"}
              </button>
            </div>
          </div>
        </section>
        <aside className="result-column">
          <RecognitionSummary result={result} />
          <div className="hint-card">
            <strong>For best results</strong>
            <p>Keep your whole hand in view. Use even lighting and a plain background.</p>
          </div>
          {!window.desktop && <p className="note">Start the Electron app for local recognition.</p>}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </aside>
      </div>
      <Scratchpad result={result} />
    </div>
  );
}
