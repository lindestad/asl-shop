import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Check,
  Files,
  Image,
  LoaderCircle,
  Plus,
  ScanLine,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { IconButton, PanelSection, Toggle } from "./components";
import { drawLandmarks, loadImage, readFiles } from "./image";
import type { BatchExport, Recognition, SourceImage } from "./types";

type BatchItem = SourceImage & {
  id: string;
  status: "queued" | "processing" | "done" | "error";
  result: Recognition | null;
  error: string | null;
};

export default function Batch({
  active,
  incoming,
  onOpen,
  notify,
}: {
  active: boolean;
  incoming: SourceImage[];
  onOpen: () => void;
  notify: (message: string) => void;
}) {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [annotate, setAnnotate] = useState(true);
  const canceled = useRef(false);
  const completed = items.filter((item) => item.status === "done");
  const errors = items.filter((item) => item.status === "error");
  const available = Boolean(window.desktop);
  function add(images: SourceImage[]) {
    setItems((current) =>
      [
        ...current,
        ...images.map((source) => ({
          ...source,
          id: crypto.randomUUID(),
          status: "queued" as const,
          result: null,
          error: null,
        })),
      ].slice(0, 100),
    );
  }
  useEffect(() => {
    if (incoming.length) add(incoming);
  }, [incoming]);
  useEffect(
    () => () => {
      canceled.current = true;
    },
    [],
  );
  async function processBatch() {
    if (!window.desktop) return;
    canceled.current = false;
    setRunning(true);
    const queue = items.filter((item) => item.status !== "done");
    for (const item of queue) {
      if (canceled.current) break;
      setItems((current) =>
        current.map((entry) =>
          entry.id === item.id ? { ...entry, status: "processing", error: null } : entry,
        ),
      );
      try {
        const image = await loadImage(item.dataUrl);
        if (image.width * image.height > 16_777_216)
          throw new Error("Image exceeds 16 megapixels.");
        const canvas = document.createElement("canvas");
        const ratio = Math.min(1, 1600 / Math.max(image.width, image.height));
        canvas.width = Math.round(image.width * ratio);
        canvas.height = Math.round(image.height * ratio);
        canvas.getContext("2d")!.drawImage(image, 0, 0, canvas.width, canvas.height);
        const result = await window.desktop.recognize(canvas.toDataURL("image/jpeg", 0.9));
        setItems((current) =>
          current.map((entry) =>
            entry.id === item.id ? { ...entry, status: "done", result } : entry,
          ),
        );
      } catch (reason) {
        setItems((current) =>
          current.map((entry) =>
            entry.id === item.id
              ? { ...entry, status: "error", error: (reason as Error).message }
              : entry,
          ),
        );
      }
    }
    setRunning(false);
  }
  async function exportBatch() {
    if (!window.desktop || !completed.length) return;
    setExporting(true);
    try {
      const output: BatchExport[] = [];
      for (const item of completed) {
        const image = await loadImage(item.dataUrl);
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(image, 0, 0);
        if (annotate && item.result)
          drawLandmarks(context, item.result.hands, canvas.width, canvas.height);
        output.push({ name: item.name, dataUrl: canvas.toDataURL(), recognition: item.result });
      }
      const destination = await window.desktop.saveBatch(output);
      if (destination)
        notify(`Exported ${output.length} images and recognition.json to ${destination}`);
    } catch (reason) {
      notify((reason as Error).message);
    } finally {
      setExporting(false);
    }
  }
  return (
    <div className="workspace" hidden={!active}>
      <section className="work-area">
        <div className="document-bar">
          <div className="document-name">
            <Files size={16} />
            <span>Image queue</span>
            <span className="small-tag">{items.length} / 100</span>
          </div>
          <div className="document-actions">
            <IconButton
              label="Clear queue"
              disabled={!items.length || running || exporting}
              onClick={() => setItems([])}
            >
              <Trash2 size={15} />
            </IconButton>
            <button
              className="button small"
              disabled={running || exporting || items.length >= 100}
              onClick={onOpen}
            >
              <Plus size={14} />
              Add images
            </button>
          </div>
        </div>
        <div
          className="batch-content"
          onDragOver={(event) => event.preventDefault()}
          onDrop={async (event) => {
            event.preventDefault();
            if (running || exporting) return;
            try {
              add(await readFiles(Array.from(event.dataTransfer.files)));
            } catch (reason) {
              notify((reason as Error).message);
            }
          }}
        >
          {!items.length ? (
            <div className="empty-batch">
              <div className="batch-art">
                <span />
                <span />
                <span>
                  <Files size={37} strokeWidth={1} />
                </span>
              </div>
              <h2>No images in queue</h2>
              <p>Add images to detect hands and export the results.</p>
              <button className="button primary" onClick={onOpen}>
                <Plus size={16} />
                Add images
                <ArrowUpRight size={15} />
              </button>
              <span className="accepted-formats">
                UP TO 100 IMAGES <span>·</span> PNG, JPG, WEBP
              </span>
            </div>
          ) : (
            <div className="batch-grid">
              {items.map((item) => (
                <article className="batch-card" key={item.id}>
                  <div className="batch-thumbnail">
                    <img src={item.dataUrl} alt={item.name} />
                    <span className={`batch-status ${item.status}`}>
                      {item.status === "done" ? (
                        <Check size={12} />
                      ) : item.status === "processing" ? (
                        <LoaderCircle size={12} className="spin" />
                      ) : item.status === "error" ? (
                        <X size={12} />
                      ) : (
                        <Image size={12} />
                      )}
                      {item.status === "done"
                        ? `${item.result?.hands.length ?? 0} hands`
                        : item.status}
                    </span>
                    <IconButton
                      label={`Remove ${item.name}`}
                      disabled={running || exporting}
                      onClick={() =>
                        setItems((current) => current.filter((entry) => entry.id !== item.id))
                      }
                    >
                      <X size={13} />
                    </IconButton>
                  </div>
                  <div className="batch-card-meta">
                    <strong title={item.name}>{item.name}</strong>
                    <span>
                      {item.error ??
                        (item.result
                          ? `${item.result.hands.reduce((sum, hand) => sum + hand.landmarks.length, 0)} landmarks extracted`
                          : "Ready for hand detection")}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
        <div className="canvas-status">
          <span>
            {completed.length} analyzed, {items.length - completed.length - errors.length} remaining
            {errors.length ? `, ${errors.length} failed` : ""}
          </span>
          <span>{running ? "Processing locally…" : ""}</span>
        </div>
      </section>
      <aside className="inspector">
        <div className="inspector-heading">
          <span className="eyebrow">BATCH WORKFLOW</span>
          <Files size={15} />
        </div>
        <div className="inspector-scroll">
          <PanelSection title="01 / Analyze">
            <p className="panel-note">
              Detect hand landmarks in every image. Letter predictions will be included once a
              classifier is available.
            </p>
            <button
              className="button primary full"
              disabled={
                !available ||
                !items.length ||
                running ||
                exporting ||
                completed.length === items.length
              }
              onClick={() => void processBatch()}
            >
              {running ? <LoaderCircle size={15} className="spin" /> : <ScanLine size={15} />}
              {running
                ? "Analyzing images…"
                : errors.length
                  ? "Retry remaining images"
                  : "Analyze all images"}
              <ArrowUpRight size={14} />
            </button>
            {running && (
              <button
                className="button full subtle"
                onClick={() => {
                  canceled.current = true;
                }}
              >
                <Square size={12} />
                Stop after current image
              </button>
            )}
            {!available && (
              <p className="panel-note">
                Batch detection and folder export are available in the desktop app.
              </p>
            )}
            <div className="batch-progress">
              <div>
                <span>Progress</span>
                <strong>
                  {completed.length} / {items.length}
                </strong>
              </div>
              <div className="confidence-track">
                <span
                  style={{
                    width: `${items.length ? (completed.length / items.length) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          </PanelSection>
          <PanelSection title="02 / Annotate">
            <Toggle label="Draw hand landmarks" checked={annotate} onChange={setAnnotate} />
            <p className="panel-note">Add landmark points and connections to exported images.</p>
          </PanelSection>
          <PanelSection title="03 / Export">
            <button
              className="button full"
              disabled={!available || !completed.length || running || exporting}
              onClick={() => void exportBatch()}
            >
              {exporting ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <ArrowDownToLine size={15} />
              )}
              {exporting ? "Preparing export…" : "Choose output folder"}
              <ArrowUpRight size={14} />
            </button>
            <p className="panel-note">
              Saves PNG images and a JSON report into a new subfolder. Original files stay intact.
            </p>
            <div className="export-summary">
              <span>
                Image format<strong>PNG</strong>
              </span>
              <span>
                Recognition report<strong>JSON</strong>
              </span>
              <span>
                Exportable images<strong>{completed.length}</strong>
              </span>
            </div>
          </PanelSection>
        </div>
      </aside>
    </div>
  );
}
