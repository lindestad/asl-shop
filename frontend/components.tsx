import { CONNECTIONS } from "./image";
import type { Recognition } from "./types";

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
      aria-hidden="true"
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
          {hand.landmarks.map((point, pointIndex) => (
            <circle key={pointIndex} cx={x(point.x)} cy={point.y} r="0.006" />
          ))}
        </g>
      ))}
    </svg>
  );
}

export function RecognitionSummary({ result }: { result: Recognition | null }) {
  return (
    <section className="result-card" aria-live="polite">
      <span className="eyebrow">RESULTAT</span>
      {result?.prediction ? (
        <>
          <strong className="result-letter">{result.prediction.letter}</strong>
          <p>{Math.round(result.prediction.confidence * 100)} % sikkerhet</p>
        </>
      ) : (
        <>
          <strong className="result-placeholder">Ingen bokstav ennå</strong>
          <p>
            {result?.hands.length
              ? "Hånd funnet. Bokstavmodellen er ikke koblet til ennå."
              : result
                ? "Ingen hånd funnet. Prøv tydeligere lys og vis hele hånden."
                : "Resultatet vises her når analysen starter."}
          </p>
        </>
      )}
      {result && (
        <div className="result-meta">
          {result.hands.length} {result.hands.length === 1 ? "hånd" : "hender"} funnet
        </div>
      )}
    </section>
  );
}
