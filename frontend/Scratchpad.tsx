import { useEffect, useRef, useState } from "react";
import { Copy, Delete, Space, Trash2 } from "lucide-react";
import type { Recognition } from "./types";

// A sign must stay recognized this long, with at least this confidence, to be typed.
const HOLD_MS = 1000;
const MIN_CONFIDENCE = 0.8;

function apply(text: string, sign: string) {
  if (sign === "space") return text + " ";
  if (sign === "del") return text.slice(0, -1);
  return text + sign;
}

export default function Scratchpad({ result }: { result: Recognition | null }) {
  const [text, setText] = useState("");
  const [held, setHeld] = useState<{ sign: string; progress: number } | null>(null);
  const hold = useRef<{ sign: string | null; since: number; done: boolean }>({
    sign: null,
    since: 0,
    done: false,
  });

  useEffect(() => {
    const prediction = result?.prediction;
    const sign =
      prediction && prediction.confidence >= MIN_CONFIDENCE && prediction.letter !== "nothing"
        ? prediction.letter
        : null;
    const now = performance.now();
    if (sign !== hold.current.sign) {
      // A sign is typed once per hold; change sign or lower the hand to repeat it.
      hold.current = { sign, since: now, done: false };
      setHeld(sign ? { sign, progress: 0 } : null);
      return;
    }
    if (!sign || hold.current.done) return;
    const progress = Math.min(1, (now - hold.current.since) / HOLD_MS);
    setHeld({ sign, progress });
    if (progress < 1) return;
    hold.current.done = true;
    setText((value) => apply(value, sign));
  }, [result]);

  return (
    <section className="scratchpad" aria-label="Scratchpad">
      <div className="scratchpad-header">
        <div>
          <span className="eyebrow">SCRATCHPAD</span>
          <p>Hold a sign for one second to type it. “space” adds a space, “del” deletes.</p>
        </div>
        <div className="hold-indicator" aria-live="polite">
          {held ? (
            <>
              <span>
                Holding <strong>{held.sign}</strong>
              </span>
              <span className="hold-bar">
                <span style={{ width: `${held.progress * 100}%` }} />
              </span>
            </>
          ) : (
            <span>Waiting for a sign</span>
          )}
        </div>
        <div className="scratchpad-actions">
          <button
            className="ghost-button"
            type="button"
            onClick={() => setText(apply(text, "space"))}
          >
            <Space size={15} /> Space
          </button>
          <button
            className="ghost-button"
            type="button"
            disabled={!text}
            onClick={() => setText(apply(text, "del"))}
          >
            <Delete size={15} /> Backspace
          </button>
          <button
            className="ghost-button"
            type="button"
            disabled={!text}
            onClick={() => void navigator.clipboard?.writeText(text)}
          >
            <Copy size={15} /> Copy
          </button>
          <button
            className="ghost-button"
            type="button"
            disabled={!text}
            onClick={() => setText("")}
          >
            <Trash2 size={15} /> Clear
          </button>
        </div>
      </div>
      <textarea
        className="scratchpad-text"
        aria-label="Text"
        placeholder="The letters you sign appear here…"
        value={text}
        onChange={(event) => setText(event.target.value)}
        spellCheck={false}
      />
    </section>
  );
}
