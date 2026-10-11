import { useState } from "react";
import { Camera, Hand, Image as ImageIcon } from "lucide-react";
import CameraView from "./Camera";
import ImageRecognition from "./ImageRecognition";

type View = "camera" | "image";

export default function App() {
  const [view, setView] = useState<View>("camera");

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand" aria-label="ASL Shop">
          <span className="brand-icon">
            <Hand size={21} strokeWidth={1.8} />
          </span>
          <span>ASL Shop</span>
        </div>
        <nav className="tabs" aria-label="Visninger">
          <button
            type="button"
            className={view === "camera" ? "active" : ""}
            aria-current={view === "camera" ? "page" : undefined}
            onClick={() => setView("camera")}
          >
            <Camera size={17} /> Live kamera
          </button>
          <button
            type="button"
            className={view === "image" ? "active" : ""}
            aria-current={view === "image" ? "page" : undefined}
            onClick={() => setView("image")}
          >
            <ImageIcon size={17} /> Bilde
          </button>
        </nav>
        <span className="app-label">ASL BOKSTAVER</span>
      </header>

      <main className="page">{view === "camera" ? <CameraView /> : <ImageRecognition />}</main>
    </div>
  );
}
