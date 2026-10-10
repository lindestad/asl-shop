import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, LoaderCircle } from "lucide-react";
import { LandmarkOverlay, RecognitionSummary } from "./components";
import type { Recognition } from "./types";

export default function CameraView() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef(0);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<Recognition | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [aspectRatio, setAspectRatio] = useState(16 / 9);

  function stop() {
    session.current++;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setRunning(false);
    setStarting(false);
    setResult(null);
  }

  useEffect(() => {
    return () => {
      session.current++;
      stream.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Kamera er ikke tilgjengelig i dette miljøet.");
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
          ? "Kameratilgang ble avslått. Gi appen tilgang i systeminnstillingene og prøv igjen."
          : name === "NotFoundError"
            ? "Fant ikke noe kamera. Koble til et kamera og prøv igjen."
            : `Kunne ikke starte kameraet: ${(reason as Error).message}`,
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

    async function analyzeFrame() {
      if (busy || !element.videoWidth) return;
      busy = true;
      try {
        const width = Math.min(640, element.videoWidth);
        canvas.width = width;
        canvas.height = Math.round(element.videoHeight * (width / element.videoWidth));
        context.drawImage(element, 0, 0, canvas.width, canvas.height);
        const recognition = await window.desktop!.recognizeFrame(
          canvas.toDataURL("image/jpeg", 0.85),
        );
        if (!canceled) {
          setResult(recognition);
          setError(null);
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
          <span className="eyebrow">SANNTID</span>
          <h1>Live kamera</h1>
          <p>Vis et tegn foran kameraet for å analysere hånden fortløpende.</p>
        </div>
        <span className={`status-pill ${running ? "live" : ""}`}>
          {running ? "Kamera på" : "Kamera av"}
        </span>
      </div>
      <div className="content-grid">
        <section className="preview-card" aria-label="Kamerabilde">
          <div className="preview-stage camera-stage" style={{ aspectRatio }}>
            <video
              ref={video}
              muted
              playsInline
              className="camera-video"
              aria-label="Live kamerabilde"
            />
            {running && <LandmarkOverlay result={result} mirrored />}
            {!running && (
              <div className="empty-state">
                <CameraOff size={34} strokeWidth={1.4} />
                <strong>Kameraet er av</strong>
                <span>Start kameraet for å se tegn her.</span>
              </div>
            )}
          </div>
          <div className="preview-footer">
            <span>
              {running ? "Håndanalyse kjører lokalt" : "Ingen video tas opp eller lagres"}
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
              {starting ? "Starter…" : running ? "Stopp kamera" : "Start kamera"}
            </button>
          </div>
        </section>
        <aside className="result-column">
          <RecognitionSummary result={result} />
          <div className="hint-card">
            <strong>For best resultat</strong>
            <p>Vis hele hånden i bildet. Bruk jevnt lys og en rolig bakgrunn.</p>
          </div>
          {!window.desktop && <p className="note">Start Electron-appen for lokal gjenkjenning.</p>}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
