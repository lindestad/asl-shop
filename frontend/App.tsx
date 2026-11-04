import { Hand } from "lucide-react";
import CameraView from "./Camera";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand" aria-label="ASL alphabet recognition">
          <span className="brand-icon">
            <Hand size={21} strokeWidth={1.8} />
          </span>
          <span>ASL alphabet recognition</span>
        </div>
        <span className="app-label">LOCAL RECOGNITION</span>
      </header>

      <main className="page">
        <CameraView />
      </main>
    </div>
  );
}
