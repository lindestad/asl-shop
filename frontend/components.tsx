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
          <p>
            {result.prediction.method === "neural"
              ? `Nevral bokstavmodell · modellscore ${Math.round(result.prediction.confidence * 100)} %`
              : `${Math.round(result.prediction.confidence * 5)} av 5 nærmeste treningseksempler`}
          </p>
        </>
      ) : (
        <>
          <strong className="result-placeholder">Ingen bokstav ennå</strong>
          <p>
            {result?.hands.length
              ? result.classifierAvailable
                ? "Hånd funnet, men ingen tydelig bokstav. Prøv en annen vinkel."
                : "Hånd funnet. Bokstavmodellen er ikke tilgjengelig."
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
      {result?.handshapeError && (
        <p className="error">Bokstavmodellen kunne ikke lastes: {result.handshapeError}</p>
      )}
    </section>
  );
}

export function FingerspellingSummary({ result }: { result: Recognition | null }) {
  const sequence = result?.fingerspelling;
  return (
    <section className="result-card" aria-live="polite" aria-label="Fingerspelling">
      <span className="eyebrow">STAVET TEKST</span>
      <p className="spelled-text">{sequence?.text || "Ingen tekst ennå"}</p>
      <p>
        {sequence?.available
          ? sequence.status === "limit"
            ? "Sekvensen er full. Start en ny sekvens."
            : sequence.provisional
              ? "Foreløpig tekst – kan endres mens du staver."
              : "Sekvensen er ferdig."
          : "Tekstmodellen er ikke tilgjengelig ennå."}
      </p>
      {sequence?.method && (
        <p>{sequence.method === "openfs" ? "OpenFS · ferdig tekst" : "Nevral sekvensmodell"}</p>
      )}
      {sequence?.contextMessage && <p>{sequence.contextMessage}</p>}
      {sequence?.refiner?.enabled && sequence.provisional && (
        <p>
          OpenFS sluttanalyse:{" "}
          {sequence.refiner.status === "ready"
            ? "klar"
            : sequence.refiner.status === "configured"
              ? "konfigurert – lastes ved fullføring"
              : "ikke tilgjengelig"}
        </p>
      )}
      {sequence?.refinementError && (
        <p className="error">
          OpenFS sluttanalyse mislyktes. Foreløpig tekst beholdes: {sequence.refinementError}
        </p>
      )}
      {sequence?.error && <p className="error">{sequence.error}</p>}
    </section>
  );
}
