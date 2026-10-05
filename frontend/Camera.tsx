import { useEffect, useRef, useState } from "react";
import { Camera, Circle, FlipHorizontal2, LoaderCircle, ScanLine, Square } from "lucide-react";
import { LandmarkOverlay, PanelSection, RecognitionPanel, Toggle } from "./components";
import type { Recognition, SourceImage } from "./types";

export default function CameraWorkspace({
  active,
  onCapture,
  notify,
}: {
  active: boolean;
  onCapture: (image: SourceImage) => void;
  notify: (message: string) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [mirrored, setMirrored] = useState(true);
  const [landmarks, setLandmarks] = useState(true);
  const [result, setResult] = useState<Recognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [size, setSize] = useState({ width: 1280, height: 720 });
  const [analyzing, setAnalyzing] = useState(false);
  const [fps, setFps] = useState(0);

  function stop() {
    ++session.current;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setRunning(false);
    setStarting(false);
    setAnalyzing(false);
    setResult(null);
    setFps(0);
  }
  useEffect(() => {
    if (!active) stop();
    return () => {
      ++session.current;
      stream.current?.getTracks().forEach((track) => track.stop());
    };
  }, [active]);

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This environment does not provide camera access. Open the desktop app.");
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
      setSize({ width: video.current!.videoWidth, height: video.current!.videoHeight });
      setRunning(true);
      setStarting(false);
    } catch (reason) {
      if (current !== session.current) return;
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = null;
      setStarting(false);
      const name = (reason as DOMException).name;
      setError(
        name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your system settings, then try again."
          : name === "NotFoundError"
            ? "No camera found. Connect a camera and try again."
            : `Could not start the camera: ${(reason as Error).message}`,
      );
    }
  }
  useEffect(() => {
    if (!running || !active || !window.desktop) return;
    const element = video.current!;
    let canceled = false;
    let busy = false;
    let frameReady = false;
    let callback = 0;
    let completedFrames = 0;
    let sampleStarted = 0;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d")!;
    setAnalyzing(true);
    async function tick() {
      busy = true;
      frameReady = false;
      try {
        const width = Math.min(640, element.videoWidth);
        const height = Math.round(element.videoHeight * (width / element.videoWidth));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        context.drawImage(element, 0, 0, width, height);
        const recognition = await window.desktop!.recognizeFrame(
          canvas.toDataURL("image/jpeg", 0.85),
        );
        if (!canceled) {
          setResult(recognition);
          setError(null);
          const now = performance.now();
          if (!sampleStarted) sampleStarted = now;
          else ++completedFrames;
          if (now - sampleStarted >= 1000) {
            setFps(Math.round((completedFrames * 1000) / (now - sampleStarted)));
            completedFrames = 0;
            sampleStarted = now;
          }
        }
      } catch (reason) {
        if (!canceled) {
          setError((reason as Error).message);
          setAnalyzing(false);
          canceled = true;
          element.cancelVideoFrameCallback(callback);
        }
      } finally {
        busy = false;
        // A newer frame may have arrived during inference. Read its current
        // pixels immediately instead of waiting for another camera interval.
        if (!canceled && frameReady) void tick();
      }
    }
    function schedule() {
      callback = element.requestVideoFrameCallback(() => {
        if (canceled) return;
        schedule();
        // Keep only the newest frame. Never queue stale frames behind inference.
        frameReady = true;
        if (!busy && element.videoWidth) void tick();
      });
    }
    schedule();
    return () => {
      canceled = true;
      element.cancelVideoFrameCallback(callback);
      setAnalyzing(false);
    };
  }, [running, active]);
  function capture() {
    if (!video.current || !running) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.current.videoWidth;
    canvas.height = video.current.videoHeight;
    const context = canvas.getContext("2d")!;
    if (mirrored) {
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
    }
    context.drawImage(video.current, 0, 0);
    onCapture({
      name: `Capture-${new Date().toISOString().replace(/[:.]/g, "-")}.png`,
      dataUrl: canvas.toDataURL(),
    });
    notify("Camera frame opened in the editor.");
    stop();
  }
  return (
    <div className="workspace" hidden={!active}>
      <section className="work-area">
        <div className="document-bar">
          <div className="document-name">
            <Camera size={16} />
            <span>Camera workspace</span>
            {running && (
              <span className="small-tag">
                {size.width} × {size.height}
              </span>
            )}
            {running && fps > 0 && (
              <span className="small-tag" aria-label="Tracking frame rate">
                {fps} FPS
              </span>
            )}
          </div>
          <span className="live-indicator">{running ? "LIVE" : "CAMERA OFF"}</span>
        </div>
        <div className="camera-stage">
          <div
            className={`camera-picture ${running ? "is-running" : ""}`}
            style={{ aspectRatio: `${size.width}/${size.height}` }}
          >
            <video
              ref={video}
              muted
              playsInline
              className={mirrored ? "mirrored" : ""}
              aria-label="Live camera preview"
            />
            {running && landmarks && <LandmarkOverlay result={result} mirrored={mirrored} />}
            {!running && (
              <div className="camera-empty">
                <div className="camera-empty-icon">
                  <Camera size={36} strokeWidth={1.1} />
                </div>
                <h2>Camera off</h2>
                <p>Start the camera to detect hands or capture an image.</p>
                <button className="button primary" disabled={starting} onClick={() => void start()}>
                  {starting ? <LoaderCircle size={16} className="spin" /> : <Camera size={16} />}
                  {starting ? "Opening camera…" : "Start camera"}
                </button>
              </div>
            )}
            <span className="camera-corner tl" />
            <span className="camera-corner tr" />
            <span className="camera-corner bl" />
            <span className="camera-corner br" />
          </div>
          {error && (
            <p className="camera-error error-message" role="alert">
              {error}
            </p>
          )}
        </div>
        <div className="camera-control-bar">
          <button className="button" disabled={!running} onClick={() => setMirrored(!mirrored)}>
            <FlipHorizontal2 size={16} />
            Mirror {mirrored ? "on" : "off"}
          </button>
          <button
            className="capture-button"
            aria-label="Capture frame"
            disabled={!running}
            onClick={capture}
          >
            <Circle size={28} fill="currentColor" strokeWidth={1} />
          </button>
          <button
            className="button"
            onClick={() => (running ? stop() : void start())}
            disabled={starting}
          >
            {running ? <Square size={12} fill="currentColor" /> : <Camera size={15} />}
            {running ? "Stop camera" : "Start camera"}
          </button>
        </div>
        <div className="canvas-status">
          <span>
            {running
              ? "Camera active, Video only, no audio"
              : "Camera access starts when you press Start"}
          </span>
          <span>Capture opens in editor</span>
        </div>
      </section>
      <aside className="inspector">
        <div className="inspector-heading">
          <span className="eyebrow">LIVE RECOGNITION</span>
          <ScanLine size={15} />
        </div>
        <div className="inspector-scroll">
          <RecognitionPanel result={result} busy={analyzing} error={null} live />
          <PanelSection title="Preview settings">
            <Toggle label="Mirror camera" checked={mirrored} onChange={setMirrored} />
            <Toggle label="Show landmarks" checked={landmarks} onChange={setLandmarks} />
            <p className="panel-note">
              Preview overlays are not included in a capture. The camera stops when you leave this
              workspace.
            </p>
          </PanelSection>
          <PanelSection title="Detection tips">
            <ol className="camera-tips">
              <li>Keep your whole hand in the frame.</li>
              <li>Use even light and a simple background.</li>
              <li>Hold the sign steady for a clear capture.</li>
            </ol>
          </PanelSection>
        </div>
      </aside>
    </div>
  );
}
