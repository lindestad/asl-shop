import { ScanLine, Check, ArrowUpRight, LoaderCircle } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { Recognition } from "./types";
import { CONNECTIONS } from "./image";

export function IconButton({
  label,
  children,
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      className={`icon-button ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
export function PanelSection({
  title,
  trailing,
  children,
}: {
  title: string;
  trailing?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel-section">
      <div className="section-heading">
        <h3>{title}</h3>
        {trailing}
      </div>
      {children}
    </section>
  );
}
export function Toggle({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
      />
      <span className="switch-track" aria-hidden="true" />
    </label>
  );
}
export function LandmarkOverlay({
  result,
  mirrored = false,
}: {
  result: Recognition | null;
  mirrored?: boolean;
}) {
  if (!result) return null;
  const x = (value: number) => (mirrored ? 1 - value : value);
  return (
    <svg
      className="landmark-overlay"
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      aria-label={`${result.hands.length} hands detected`}
    >
      {result.hands.map((hand, index) => (
        <g key={index}>
          {CONNECTIONS.map(([a, b]) => (
            <line
              key={`${a}-${b}`}
              x1={x(hand.landmarks[a].x)}
              y1={hand.landmarks[a].y}
              x2={x(hand.landmarks[b].x)}
              y2={hand.landmarks[b].y}
            />
          ))}
          {hand.landmarks.map((point, i) => (
            <circle key={i} cx={x(point.x)} cy={point.y} r="0.006" />
          ))}
        </g>
      ))}
    </svg>
  );
}
export function RecognitionPanel({
  result,
  busy,
  error,
  onAnalyze,
  canAnalyze = true,
  live = false,
}: {
  result: Recognition | null;
  busy: boolean;
  error: string | null;
  onAnalyze?: () => void;
  canAnalyze?: boolean;
  live?: boolean;
}) {
  const available = Boolean(window.desktop);
  return (
    <>
      <PanelSection title="Sign recognition" trailing={<ScanLine size={15} />}>
        {result?.prediction ? (
          <div className="prediction-card">
            <span className="eyebrow">DETECTED LETTER</span>
            <strong className="prediction-letter">{result.prediction.letter}</strong>
            <span className="prediction-caption">
              {Math.round(result.prediction.confidence * 100)}% letter confidence
            </span>
          </div>
        ) : (
          <p className="panel-note">Letter classifier unavailable</p>
        )}
        {onAnalyze && (
          <button
            className="button full"
            disabled={!available || !canAnalyze || busy}
            onClick={onAnalyze}
          >
            {busy ? <LoaderCircle className="spin" size={15} /> : <ScanLine size={15} />}
            {busy ? "Analyzing…" : "Analyze image"}
            <ArrowUpRight size={14} />
          </button>
        )}
        {!available && (
          <p className="panel-note">Open the desktop app to use local hand detection.</p>
        )}
        {error && (
          <p className="error-message" role="alert">
            {error}
          </p>
        )}
      </PanelSection>
      <PanelSection
        title="Hand landmarks"
        trailing={<span className="small-tag">{result?.hands.length ?? 0} HANDS</span>}
      >
        {result?.hands.length ? (
          result.hands.map((hand, index) => (
            <div className="hand-result" key={index}>
              <div className="hand-result-title">
                <span>
                  <Check size={13} />
                  {hand.hand} hand
                </span>
                <span>{Math.round(hand.score * 100)}%</span>
              </div>
              <div className="confidence-track">
                <span style={{ width: `${hand.score * 100}%` }} />
              </div>
              <p>21 landmarks, 63 features, hand confidence</p>
            </div>
          ))
        ) : (
          <div className="landmark-empty">
            <ScanLine size={22} strokeWidth={1.2} />
            <span>
              {busy
                ? "Looking for hands…"
                : result
                  ? "No hands found in this frame"
                  : live
                    ? "Start your camera to detect hands"
                    : "Analyze an image to see its landmarks"}
            </span>
          </div>
        )}
      </PanelSection>
    </>
  );
}
