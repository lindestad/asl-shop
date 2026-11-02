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
      <span className="eyebrow">RESULT</span>
      {result?.prediction ? (
        <>
          <strong className="result-letter">{result.prediction.letter}</strong>
          <p>{Math.round(result.prediction.confidence * 5)} of 5 nearest training examples</p>
        </>
      ) : (
        <>
          <strong className="result-placeholder">No letter yet</strong>
          <p>
            {result?.hands.length
              ? result.classifierAvailable
                ? "Hand found, but no clear letter. Try another angle."
                : "Hand found. The letter model is not available."
              : result
                ? "No hand found. Try brighter light and show your whole hand."
                : "The result appears here once analysis starts."}
          </p>
        </>
      )}
      {result && (
        <div className="result-meta">
          {result.hands.length} {result.hands.length === 1 ? "hand" : "hands"} found
        </div>
      )}
    </section>
  );
}
