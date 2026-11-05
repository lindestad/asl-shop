import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Camera, CameraOff, Clapperboard, LoaderCircle } from "lucide-react";
import { FingerspellingSummary, LandmarkOverlay, RecognitionSummary } from "./components";
import Scratchpad from "./Scratchpad";
import type { DemoClip, DemoClipInfo, Recognition } from "./types";

export default function CameraView() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const textSession = useRef(crypto.randomUUID());
  const finishRequested = useRef(false);
  const replayingRef = useRef(false);
  const frameBusy = useRef(false);
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

  async function sendReplayFrame(
    desktop: NonNullable<Window["desktop"]>,
    sessionId: string,
    frames: DemoClip["frames"],
    i: number,
  ) {
    const recognition = await desktop.recognizeLandmarks(frames[i].hands, {
      sessionId,
      timestampMs: frames[i].t,
      finish: i === frames.length - 1,
    });
    if (sessionId === textSession.current) setResult(recognition);
  }

  async function playClip() {
    const desktop = window.desktop;
    if (!desktop) return;
    try {
      const clip = await desktop.demoClip(demoIndex);
      newText();
      const sessionId = textSession.current;
      const frames = clip.frames;
      const total = frames.length;
      replayingRef.current = true;
      setReplaying(true);
      setError(null);

      if (!clip.video) {
        for (let i = 0; i < total && replayingRef.current; i++) {
          await sendReplayFrame(desktop, sessionId, frames, i);
          if (i + 1 < total) await new Promise((tick) => setTimeout(tick, 33));
        }
        return;
      }

      // A demo clip *is* the camera: the same frame loop, worker session and
      // finish semantics as a live signing run, fed from dataset footage instead
      // of getUserMedia. MediaPipe detects the displayed pixels, so overlays are
      // the landmarks the pipeline actually consumed.
      if (stream.current || running) stop();
      const element = video.current!;
      element.srcObject = null;
      element.src = `demo-media://clips/${clip.video}`;
      await new Promise<void>((resolve, reject) => {
        element.addEventListener("loadedmetadata", () => resolve(), { once: true });
        element.addEventListener("error", () => reject(new Error("Demo footage failed to load.")), {
          once: true,
        });
      });
      setAspectRatio(element.videoWidth / element.videoHeight);
      setRunning(true);
      await element.play();
      await new Promise<void>((resolve) => {
        element.addEventListener("ended", () => resolve(), { once: true });
        element.addEventListener("pause", () => resolve(), { once: true });
      });
      if (!replayingRef.current) return; // the stop button ended the replay
      // The clip is over and no further video frames will be scheduled, so the
      // finish flag goes out on one last snapshot of the final frame, after the
      // loop's last in-flight analysis has landed.
      while (frameBusy.current) await new Promise((tick) => setTimeout(tick, 60));
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d")!;
      const width = Math.min(640, element.videoWidth);
      canvas.width = width;
      canvas.height = Math.round(element.videoHeight * (width / element.videoWidth));
      context.drawImage(element, 0, 0, canvas.width, canvas.height);
      const segmentId = textSession.current;
      const recognition = await desktop.recognizeFrame(canvas.toDataURL("image/jpeg", 0.85), {
        sessionId: segmentId,
        timestampMs: performance.now(),
        finish: true,
      });
      if (segmentId === textSession.current) setResult(recognition);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      const element = video.current;
      if (element && !element.srcObject) {
        element.pause();
        if (element.getAttribute("src")) {
          element.removeAttribute("src");
          element.load();
        }
      }
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
    if (video.current) {
      video.current.pause();
      video.current.srcObject = null;
      if (video.current.getAttribute("src")) {
        video.current.removeAttribute("src");
        video.current.load();
      }
    }
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
    let callback = 0;
    let completedFrames = 0;
    let sampleStarted = 0;

    async function analyzeFrame() {
      if (frameBusy.current || !element.videoWidth) return;
      frameBusy.current = true;
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
        frameBusy.current = false;
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
                crossOrigin="anonymous"
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
                {clip.video ? " · footage" : ""}
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
