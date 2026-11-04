import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Camera, CameraOff, Clapperboard, LoaderCircle } from "lucide-react";
import { FingerspellingSummary, LandmarkOverlay, RecognitionSummary } from "./components";
import Scratchpad from "./Scratchpad";
import type { DemoClipInfo, Recognition } from "./types";

export default function CameraView() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const textSession = useRef(crypto.randomUUID());
  const finishRequested = useRef(false);
  const replayingRef = useRef(false);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<Recognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [aspectRatio, setAspectRatio] = useState(16 / 9);
  const [fps, setFps] = useState(0);
  const [demoClips, setDemoClips] = useState<DemoClipInfo[]>([]);
  const [demoIndex, setDemoIndex] = useState(0);
  const [replaying, setReplaying] = useState(false);

  useEffect(() => {
    if (!window.desktop) return;
    window.desktop
      .demoList()
      .then(setDemoClips)
      .catch(() => setDemoClips([]));
  }, []);

  async function playClip() {
    const desktop = window.desktop;
    if (!desktop) return;
    try {
      const clip = await desktop.demoClip(demoIndex);
      newText();
      const sessionId = textSession.current;
      replayingRef.current = true;
      setReplaying(true);
      setError(null);
      for (let i = 0; i < clip.frames.length && replayingRef.current; i++) {
        const frame = clip.frames[i];
        const recognition = await desktop.recognizeLandmarks(frame.hands, {
          sessionId,
          timestampMs: frame.t,
          finish: i === clip.frames.length - 1,
        });
        if (sessionId === textSession.current) setResult(recognition);
        if (i + 1 < clip.frames.length) await new Promise((tick) => setTimeout(tick, 33));
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      replayingRef.current = false;
      setReplaying(false);
    }
  }

  function newText() {
    textSession.current = crypto.randomUUID();
    finishRequested.current = false;
    setResult(null);
  }

  function stop() {
    session.current++;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setRunning(false);
    setStarting(false);
    setResult(null);
    setFps(0);
    replayingRef.current = false;
    setReplaying(false);
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
    newText();
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
    let busy = false;
    let callback = 0;
    let completedFrames = 0;
    let sampleStarted = 0;

    async function analyzeFrame() {
      if (busy || !element.videoWidth) return;
      busy = true;
      try {
        const width = Math.min(640, element.videoWidth);
        canvas.width = width;
        canvas.height = Math.round(element.videoHeight * (width / element.videoWidth));
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        const segmentId = textSession.current;
        const recognition = await window.desktop!.recognizeFrame(
          canvas.toDataURL("image/jpeg", 0.85),
          { sessionId: segmentId, timestampMs: performance.now(), finish: finishRequested.current },
        );
        if (!canceled && segmentId === textSession.current) {
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
        busy = false;
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
  }, [running]);

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
              {(running || replaying) && (
                <LandmarkOverlay result={result} mirrored={running && !replaying} />
              )}
            </div>
            {!running && !replaying && (
              <div className="empty-state">
                <CameraOff size={34} strokeWidth={1.4} />
                <strong>The camera is off</strong>
                <span>Start the camera to see your signs here.</span>
              </div>
            )}
            {replaying && (
              <div className="empty-state">
                <Clapperboard size={34} strokeWidth={1.4} />
                <strong>Replaying a dataset clip</strong>
                <span>Landmark frames drive the same live pipeline, no camera involved.</span>
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
        </section>
        <aside className="result-column">
          <RecognitionSummary result={result} />
          <FingerspellingSummary result={result} />
          {running && result?.fingerspelling?.available && (
            <div className="sequence-controls">
              <button
                type="button"
                className="primary-button"
                disabled={result.fingerspelling.status === "finished"}
                onClick={() => {
                  finishRequested.current = true;
                }}
              >
                Fullfør tekst
              </button>
              <button type="button" className="primary-button" onClick={newText}>
                Ny sekvens
              </button>
            </div>
          )}
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
      {window.desktop && demoClips.length > 0 && (
        <section className="replay-card" aria-label="Dataset replay">
          <div className="replay-copy">
            <span className="eyebrow">DATASET REPLAY</span>
            <p>
              No camera? Replay landmark frames from dataset clips through the same live and
              final-text pipeline.
            </p>
          </div>
          <select
            aria-label="Replay clip"
            value={demoIndex}
            disabled={replaying}
            onChange={(event) => setDemoIndex(Number(event.target.value))}
          >
            {demoClips.map((clip, position) => (
              <option key={clip.index} value={position}>
                clip {position + 1} · {(clip.durationMs / 1000).toFixed(1)} s · {clip.text}
              </option>
            ))}
          </select>
          <button
            className="primary-button"
            type="button"
            disabled={replaying}
            onClick={() => void playClip()}
          >
            {replaying ? <LoaderCircle className="spin" size={17} /> : <Clapperboard size={17} />}
            {replaying ? "Replaying…" : "Play clip"}
          </button>
        </section>
      )}
      <Scratchpad result={result} />
    </div>
  );
}
