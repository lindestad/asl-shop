import { useRef, useState } from "react";
import { ImagePlus, LoaderCircle } from "lucide-react";
import { LandmarkOverlay, RecognitionSummary } from "./components";
import { readImage } from "./image";
import type { Recognition, SourceImage } from "./types";

export default function ImageRecognition() {
  const input = useRef<HTMLInputElement>(null);
  const request = useRef(0);
  const [image, setImage] = useState<SourceImage | null>(null);
  const [result, setResult] = useState<Recognition | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze(selected: SourceImage) {
    const current = ++request.current;
    setImage(selected);
    setResult(null);
    setError(null);
    if (!window.desktop) return;
    setBusy(true);
    try {
      const recognition = await window.desktop.recognize(selected.dataUrl);
      if (current === request.current) setResult(recognition);
    } catch (reason) {
      if (current === request.current) setError((reason as Error).message);
    } finally {
      if (current === request.current) setBusy(false);
    }
  }

  async function chooseImage() {
    if (!window.desktop) {
      input.current?.click();
      return;
    }
    try {
      const selected = await window.desktop.openImage();
      if (selected) await analyze(selected);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  return (
    <div className="view">
      <div className="view-heading">
        <div>
          <span className="eyebrow">STILLBILDE</span>
          <h1>Gjenkjenn fra bilde</h1>
          <p>Åpne ett bilde. Hånden analyseres automatisk.</p>
        </div>
        <button className="primary-button" type="button" onClick={() => void chooseImage()}>
          <ImagePlus size={17} /> Velg bilde
        </button>
      </div>

      <div className="content-grid">
        <section className="preview-card" aria-label="Valgt bilde">
          <div className="preview-stage image-stage">
            {image ? (
              <div className="preview-image-frame">
                <img src={image.dataUrl} alt={image.name} />
                <LandmarkOverlay result={result} />
              </div>
            ) : (
              <div className="empty-state">
                <ImagePlus size={34} strokeWidth={1.4} />
                <strong>Intet bilde valgt</strong>
                <span>Velg et bilde av et håndtegn for å begynne.</span>
              </div>
            )}
          </div>
          <div className="preview-footer">
            <span className="filename">{image?.name ?? "PNG, JPEG eller WebP · maks 16 MB"}</span>
            {busy && (
              <span className="busy-label">
                <LoaderCircle className="spin" size={16} /> Analyserer…
              </span>
            )}
          </div>
        </section>
        <aside className="result-column">
          <RecognitionSummary result={result} />
          <div className="hint-card">
            <strong>Om bildeanalysen</strong>
            <p>Bildet behandles lokalt i appen. Hele hånden bør være synlig.</p>
          </div>
          {!window.desktop && <p className="note">Start Electron-appen for lokal gjenkjenning.</p>}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </aside>
      </div>
      <input
        ref={input}
        data-testid="image-input"
        className="visually-hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file) return;
          try {
            await analyze(await readImage(file));
          } catch (reason) {
            setError((reason as Error).message);
          }
        }}
      />
    </div>
  );
}
