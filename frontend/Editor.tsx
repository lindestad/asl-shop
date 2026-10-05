import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Brush,
  Camera,
  ChevronDown,
  Circle,
  Crop,
  Eraser,
  Expand,
  FlipHorizontal2,
  FlipVertical2,
  Image,
  MousePointer2,
  Pipette,
  Plus,
  Redo2,
  RotateCcw,
  RotateCw,
  ScanLine,
  SlidersHorizontal,
  Square,
  Type,
  Undo2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { IconButton, LandmarkOverlay, PanelSection, RecognitionPanel, Toggle } from "./components";
import { applyFilter, loadImage, readFiles, saveImage } from "./image";
import type { Filter } from "./image";
import type { Recognition, SourceImage } from "./types";

type DocumentImage = SourceImage & { width: number; height: number };
type Tool = "select" | "brush" | "eraser" | "rectangle" | "ellipse" | "text" | "picker";
type Selection = { x: number; y: number; width: number; height: number };
const tools = [
  { id: "select", name: "Rectangular selection", icon: MousePointer2 },
  { id: "brush", name: "Brush", icon: Brush },
  { id: "eraser", name: "Eraser", icon: Eraser },
  { id: "rectangle", name: "Rectangle", icon: Square },
  { id: "ellipse", name: "Ellipse", icon: Circle },
  { id: "text", name: "Text", icon: Type },
  { id: "picker", name: "Color picker", icon: Pipette },
] as const;

export default function Editor({
  active,
  incoming,
  newRequest,
  onOpen,
  onCamera,
  notify,
}: {
  active: boolean;
  incoming: SourceImage | null;
  newRequest: number;
  onOpen: () => void;
  onCamera: () => void;
  notify: (message: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<DocumentImage | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [color, setColor] = useState("#171717");
  const [brushSize, setBrushSize] = useState(12);
  const [shapeFill, setShapeFill] = useState(false);
  const [text, setText] = useState("Your text");
  const [zoom, setZoom] = useState(100);
  const [available, setAvailable] = useState({ width: 600, height: 500 });
  const [selection, setSelection] = useState<Selection | null>(null);
  const gesture = useRef<{ x: number; y: number; snapshot: ImageData } | null>(null);
  const history = useRef<DocumentImage[]>([]);
  const future = useRef<DocumentImage[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [panel, setPanel] = useState<"adjust" | "recognize">("adjust");
  const [width, setWidth] = useState(1200);
  const [height, setHeight] = useState(800);
  const [threshold, setThreshold] = useState(128);
  const [newDialog, setNewDialog] = useState(false);
  const [result, setResult] = useState<Recognition | null>(null);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const fileMenu = useRef<HTMLDetailsElement>(null);

  async function load(source: SourceImage) {
    const current = ++generation.current;
    try {
      const decoded = await loadImage(source.dataUrl);
      if (decoded.width * decoded.height > 16_777_216)
        throw new Error("Choose an image smaller than 16 megapixels.");
      if (generation.current !== current) return;
      setImage({ ...source, width: decoded.width, height: decoded.height });
      setWidth(decoded.width);
      setHeight(decoded.height);
      setZoom(100);
      setSelection(null);
      setResult(null);
      setError(null);
      history.current = [];
      future.current = [];
      setHistoryVersion((value) => value + 1);
    } catch (reason) {
      notify((reason as Error).message);
    }
  }
  useEffect(() => {
    if (incoming) void load(incoming);
  }, [incoming]);
  useEffect(() => {
    if (newRequest) setNewDialog(true);
  }, [newRequest]);
  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) =>
      setAvailable({
        width: Math.max(100, entry.contentRect.width - 100),
        height: Math.max(100, entry.contentRect.height - 100),
      }),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!image) return;
    let canceled = false;
    void loadImage(image.dataUrl).then((decoded) => {
      if (canceled || !canvasRef.current) return;
      const canvas = canvasRef.current;
      canvas.width = image.width;
      canvas.height = image.height;
      canvas.getContext("2d", { willReadFrequently: true })!.drawImage(decoded, 0, 0);
    });
    return () => {
      canceled = true;
    };
  }, [image]);

  function commit() {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    history.current = [...history.current.slice(-9), image];
    future.current = [];
    setImage({
      name: image.name,
      dataUrl: canvas.toDataURL(),
      width: canvas.width,
      height: canvas.height,
    });
    setWidth(canvas.width);
    setHeight(canvas.height);
    setSelection(null);
    setResult(null);
    setError(null);
    ++generation.current;
    setHistoryVersion((value) => value + 1);
  }
  function restore(direction: "undo" | "redo") {
    if (!image) return;
    const source = direction === "undo" ? history.current : future.current;
    const destination = direction === "undo" ? future.current : history.current;
    const previous = source.pop();
    if (!previous) return;
    destination.push(image);
    setImage(previous);
    setWidth(previous.width);
    setHeight(previous.height);
    setSelection(null);
    setResult(null);
    ++generation.current;
    setHistoryVersion((value) => value + 1);
  }
  useEffect(() => {
    function handle(event: KeyboardEvent) {
      if (
        !active ||
        !image ||
        (event.target instanceof HTMLElement &&
          ["INPUT", "TEXTAREA"].includes(event.target.tagName))
      )
        return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        restore(event.shiftKey ? "redo" : "undo");
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void exportImage();
      }
    }
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [active, image, historyVersion]);

  function createCanvas() {
    if (!validDimensions(width, height)) {
      notify("Use dimensions from 1 to 4096 px, up to 16 megapixels.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    void load({ name: "Untitled.png", dataUrl: canvas.toDataURL() });
    setNewDialog(false);
  }
  function transform(operation: "left" | "right" | "horizontal" | "vertical" | "resize" | "crop") {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    if (operation === "resize" && !validDimensions(width, height)) {
      notify("Use dimensions from 1 to 4096 px, up to 16 megapixels.");
      return;
    }
    if (operation === "crop" && (!selection || selection.width < 1 || selection.height < 1)) return;
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    copy.getContext("2d")!.drawImage(canvas, 0, 0);
    const rotates = operation === "left" || operation === "right";
    canvas.width = rotates
      ? copy.height
      : operation === "resize"
        ? width
        : operation === "crop"
          ? Math.round(selection!.width)
          : copy.width;
    canvas.height = rotates
      ? copy.width
      : operation === "resize"
        ? height
        : operation === "crop"
          ? Math.round(selection!.height)
          : copy.height;
    const context = canvas.getContext("2d", { willReadFrequently: true })!;
    context.save();
    if (rotates) {
      context.translate(canvas.width / 2, canvas.height / 2);
      context.rotate(((operation === "right" ? 1 : -1) * Math.PI) / 2);
      context.drawImage(copy, -copy.width / 2, -copy.height / 2);
    } else if (operation === "horizontal" || operation === "vertical") {
      context.translate(
        operation === "horizontal" ? canvas.width : 0,
        operation === "vertical" ? canvas.height : 0,
      );
      context.scale(operation === "horizontal" ? -1 : 1, operation === "vertical" ? -1 : 1);
      context.drawImage(copy, 0, 0);
    } else if (operation === "crop")
      context.drawImage(
        copy,
        selection!.x,
        selection!.y,
        selection!.width,
        selection!.height,
        0,
        0,
        canvas.width,
        canvas.height,
      );
    else context.drawImage(copy, 0, 0, canvas.width, canvas.height);
    context.restore();
    commit();
  }
  async function exportImage() {
    if (!image) return;
    try {
      if (
        await saveImage({
          name: image.name.replace(/\.[^.]+$/, "") + ".png",
          dataUrl: canvasRef.current!.toDataURL(),
        })
      )
        notify("Image exported.");
    } catch (reason) {
      notify((reason as Error).message);
    }
  }
  async function analyze() {
    if (!image || !window.desktop) return;
    const current = generation.current;
    setBusy(true);
    setError(null);
    try {
      const recognition = await window.desktop.recognize(
        canvasRef.current!.toDataURL("image/jpeg", 0.92),
      );
      if (current === generation.current) setResult(recognition);
    } catch (reason) {
      if (current === generation.current) setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function position(event: PointerEvent<HTMLCanvasElement>) {
    const canvas = event.currentTarget,
      rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(canvas.width, ((event.clientX - rect.left) * canvas.width) / rect.width),
      ),
      y: Math.max(
        0,
        Math.min(canvas.height, ((event.clientY - rect.top) * canvas.height) / rect.height),
      ),
    };
  }
  function pointerDown(event: PointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0 || !image) return;
    const canvas = event.currentTarget,
      context = canvas.getContext("2d", { willReadFrequently: true })!,
      point = position(event);
    if (tool === "picker") {
      const rgba = context.getImageData(
        Math.min(canvas.width - 1, Math.floor(point.x)),
        Math.min(canvas.height - 1, Math.floor(point.y)),
        1,
        1,
      ).data;
      setColor(
        "#" +
          Array.from(rgba.slice(0, 3))
            .map((value) => value.toString(16).padStart(2, "0"))
            .join(""),
      );
      return;
    }
    if (tool === "text") {
      context.fillStyle = color;
      context.font = `${Math.max(16, brushSize * 3)}px sans-serif`;
      context.fillText(text, point.x, point.y);
      commit();
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    gesture.current = {
      ...point,
      snapshot: context.getImageData(0, 0, canvas.width, canvas.height),
    };
    setSelection(null);
    context.beginPath();
    context.moveTo(point.x, point.y);
    if (tool === "brush" || tool === "eraser") {
      context.save();
      context.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
      context.fillStyle = color;
      context.beginPath();
      context.arc(point.x, point.y, brushSize / 2, 0, Math.PI * 2);
      context.fill();
      context.restore();
      context.beginPath();
      context.moveTo(point.x, point.y);
    }
  }
  function pointerMove(event: PointerEvent<HTMLCanvasElement>) {
    const start = gesture.current;
    if (!start) return;
    const point = position(event),
      canvas = event.currentTarget,
      context = canvas.getContext("2d", { willReadFrequently: true })!;
    const rect = {
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      width: Math.abs(point.x - start.x),
      height: Math.abs(point.y - start.y),
    };
    if (tool === "select") {
      setSelection(rect);
      return;
    }
    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = brushSize;
    context.lineCap = "round";
    context.lineJoin = "round";
    if (tool === "brush" || tool === "eraser") {
      context.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
      context.lineTo(point.x, point.y);
      context.stroke();
    } else {
      context.putImageData(start.snapshot, 0, 0);
      context.beginPath();
      if (tool === "rectangle") context.rect(rect.x, rect.y, rect.width, rect.height);
      else
        context.ellipse(
          rect.x + rect.width / 2,
          rect.y + rect.height / 2,
          rect.width / 2,
          rect.height / 2,
          0,
          0,
          Math.PI * 2,
        );
      if (shapeFill) context.fill();
      context.stroke();
    }
    context.restore();
  }
  function pointerUp() {
    if (!gesture.current) return;
    gesture.current = null;
    if (tool !== "select") commit();
  }
  const scale = image
    ? (Math.min(available.width / image.width, available.height / image.height, 1) * zoom) / 100
    : 1;

  return (
    <div className="workspace editor-workspace" hidden={!active}>
      <section className="work-area">
        <div className="document-bar">
          <div className="document-name">
            <span className="document-icon">
              <Image size={14} />
            </span>
            <span>{image?.name ?? "Untitled workspace"}</span>
            {image && (
              <span className="small-tag">
                {image.width} × {image.height}
              </span>
            )}
          </div>
          <div className="document-actions">
            <details ref={fileMenu} className="file-menu">
              <summary>
                File
                <ChevronDown size={12} />
              </summary>
              <div className="menu-popover">
                <button
                  onClick={() => {
                    setNewDialog(true);
                    fileMenu.current!.open = false;
                  }}
                >
                  New canvas
                </button>
                <button
                  onClick={() => {
                    onOpen();
                    fileMenu.current!.open = false;
                  }}
                >
                  Open image
                </button>
                <button
                  disabled={!image}
                  onClick={() => {
                    void exportImage();
                    fileMenu.current!.open = false;
                  }}
                >
                  Export PNG
                </button>
              </div>
            </details>
            <span className="toolbar-divider" />
            <IconButton
              label="Undo"
              disabled={!history.current.length}
              onClick={() => restore("undo")}
            >
              <Undo2 size={16} />
            </IconButton>
            <IconButton
              label="Redo"
              disabled={!future.current.length}
              onClick={() => restore("redo")}
            >
              <Redo2 size={16} />
            </IconButton>
            <span className="toolbar-divider" />
            <button className="button small" disabled={!image} onClick={() => void exportImage()}>
              <ArrowDownToLine size={14} />
              Export
            </button>
          </div>
        </div>
        <div className="canvas-toolbar">
          <div className="tool-group" role="toolbar" aria-label="Drawing tools">
            {tools.map(({ id, name, icon: Icon }) => (
              <IconButton
                key={id}
                label={name}
                disabled={!image}
                className={tool === id ? "selected" : ""}
                aria-pressed={tool === id}
                onClick={() => setTool(id)}
              >
                <Icon size={17} strokeWidth={1.5} />
              </IconButton>
            ))}
            <span className="toolbar-divider" />
            <IconButton
              label="Crop selection"
              disabled={!selection || selection.width < 1 || selection.height < 1}
              onClick={() => transform("crop")}
            >
              <Crop size={17} />
            </IconButton>
          </div>
          <div className="zoom-controls">
            <IconButton
              label="Zoom out"
              disabled={!image || zoom <= 25}
              onClick={() => setZoom(Math.max(25, zoom - 25))}
            >
              <ZoomOut size={15} />
            </IconButton>
            <button
              className="zoom-value"
              disabled={!image}
              onClick={() => setZoom(100)}
              title="Fit image to workspace"
            >
              {zoom}%
            </button>
            <IconButton
              label="Zoom in"
              disabled={!image || zoom >= 300}
              onClick={() => setZoom(Math.min(300, zoom + 25))}
            >
              <ZoomIn size={15} />
            </IconButton>
            <IconButton label="Fit to workspace" disabled={!image} onClick={() => setZoom(100)}>
              <Expand size={15} />
            </IconButton>
          </div>
        </div>
        <div
          ref={viewportRef}
          className={`canvas-viewport ${image ? "has-image" : ""}`}
          onDragOver={(event) => event.preventDefault()}
          onDrop={async (event) => {
            event.preventDefault();
            try {
              const images = await readFiles(Array.from(event.dataTransfer.files));
              if (images[0]) await load(images[0]);
            } catch (reason) {
              notify((reason as Error).message);
            }
          }}
        >
          <div className="canvas-inner">
            {!image ? (
              <div className="empty-editor">
                <div className="empty-art">
                  <span className="corner top-left" />
                  <span className="corner top-right" />
                  <span className="corner bottom-left" />
                  <span className="corner bottom-right" />
                  <Image size={45} strokeWidth={1} />
                  <span className="art-plus">
                    <Plus size={17} />
                  </span>
                </div>
                <h2>Drop an image here</h2>
                <p>PNG, JPEG, or WebP, Up to 30 MB and 16 megapixels</p>
                <button className="button primary" onClick={onOpen}>
                  <Plus size={16} />
                  Open image
                  <ArrowUpRight size={15} />
                </button>
                <div className="empty-secondary">
                  <button onClick={() => setNewDialog(true)}>
                    <Square size={14} />
                    Create a canvas
                  </button>
                  <span />
                  <button onClick={onCamera}>
                    <Camera size={15} />
                    Use your camera
                  </button>
                </div>
              </div>
            ) : (
              <div
                className="image-frame"
                style={{ width: image.width * scale, height: image.height * scale }}
              >
                <canvas
                  ref={canvasRef}
                  data-testid="editor-canvas"
                  aria-label="Image editing canvas"
                  style={{
                    width: image.width * scale,
                    height: image.height * scale,
                    cursor:
                      tool === "select" ? "crosshair" : tool === "text" ? "text" : "crosshair",
                  }}
                  onPointerDown={pointerDown}
                  onPointerMove={pointerMove}
                  onPointerUp={pointerUp}
                  onPointerCancel={() => {
                    if (gesture.current)
                      canvasRef
                        .current!.getContext("2d")!
                        .putImageData(gesture.current.snapshot, 0, 0);
                    gesture.current = null;
                    setSelection(null);
                  }}
                />
                {showLandmarks && <LandmarkOverlay result={result} />}
                {selection && (
                  <div
                    className="selection-box"
                    style={{
                      left: selection.x * scale,
                      top: selection.y * scale,
                      width: selection.width * scale,
                      height: selection.height * scale,
                    }}
                  />
                )}
              </div>
            )}
          </div>
        </div>
        <div className="canvas-status">
          <span>
            {image ? `${image.width} × ${image.height} px, RGB` : "No image selected"}
            {selection &&
              `, Selection ${Math.round(selection.width)} × ${Math.round(selection.height)}`}
          </span>
          <span>{image ? tools.find((item) => item.id === tool)?.name : ""}</span>
        </div>
      </section>
      <aside className="inspector">
        <div className="inspector-heading">
          <span className="eyebrow">INSPECTOR</span>
          <SlidersHorizontal size={15} />
        </div>
        <div className="panel-tabs">
          <button className={panel === "adjust" ? "active" : ""} onClick={() => setPanel("adjust")}>
            <SlidersHorizontal size={14} />
            Adjust
          </button>
          <button
            className={panel === "recognize" ? "active" : ""}
            onClick={() => setPanel("recognize")}
          >
            <ScanLine size={14} />
            Recognize
          </button>
        </div>
        <div className="inspector-scroll">
          {panel === "adjust" ? (
            <>
              <PanelSection title="Image properties">
                <div className="dimension-fields">
                  <label>
                    Width
                    <span>
                      <input
                        aria-label="Image width"
                        type="number"
                        min="1"
                        max="4096"
                        value={image ? width : ""}
                        placeholder="—"
                        disabled={!image}
                        onChange={(event) => setWidth(Number(event.target.value))}
                      />
                      <small>px</small>
                    </span>
                  </label>
                  <span className="dimension-cross">×</span>
                  <label>
                    Height
                    <span>
                      <input
                        aria-label="Image height"
                        type="number"
                        min="1"
                        max="4096"
                        value={image ? height : ""}
                        placeholder="—"
                        disabled={!image}
                        onChange={(event) => setHeight(Number(event.target.value))}
                      />
                      <small>px</small>
                    </span>
                  </label>
                </div>
                <button
                  className="button full subtle"
                  disabled={!image}
                  onClick={() => transform("resize")}
                >
                  Resize image
                  <Expand size={14} />
                </button>
              </PanelSection>
              <PanelSection title="Transform">
                <div className="transform-grid">
                  {[
                    { op: "left", label: "Rotate left", icon: RotateCcw },
                    { op: "right", label: "Rotate right", icon: RotateCw },
                    { op: "horizontal", label: "Flip horizontal", icon: FlipHorizontal2 },
                    { op: "vertical", label: "Flip vertical", icon: FlipVertical2 },
                  ].map(({ op, label, icon: Icon }) => (
                    <button
                      key={op}
                      disabled={!image}
                      onClick={() => transform(op as "left" | "right" | "horizontal" | "vertical")}
                    >
                      <Icon size={18} strokeWidth={1.4} />
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
              </PanelSection>
              <PanelSection title="Filters">
                <div className="filter-grid">
                  {[
                    { id: "grayscale", title: "Grayscale", className: "grayscale" },
                    { id: "blur", title: "Gaussian blur", className: "blur" },
                    { id: "sobel", title: "Sobel edges", className: "sobel" },
                    { id: "binary", title: "Binary", className: "binary" },
                  ].map((filter) => (
                    <button
                      key={filter.id}
                      disabled={!image}
                      onClick={() => {
                        applyFilter(canvasRef.current!, filter.id as Filter, threshold);
                        commit();
                      }}
                    >
                      <span className={`filter-preview ${filter.className}`}>
                        <span />
                      </span>
                      <span>{filter.title}</span>
                    </button>
                  ))}
                </div>
                <label className="range-label">
                  Binary threshold<span>{threshold}</span>
                  <input
                    aria-label="Binary threshold"
                    type="range"
                    min="0"
                    max="255"
                    value={threshold}
                    onChange={(event) => setThreshold(Number(event.target.value))}
                    disabled={!image}
                  />
                </label>
              </PanelSection>
              <PanelSection title="Tool settings">
                <label className="color-control">
                  <span>Color</span>
                  <input
                    aria-label="Drawing color"
                    type="color"
                    value={color}
                    onChange={(event) => setColor(event.target.value)}
                  />
                  <code>{color.toUpperCase()}</code>
                </label>
                <label className="range-label">
                  Brush / outline size<span>{brushSize} px</span>
                  <input
                    aria-label="Brush size"
                    type="range"
                    min="1"
                    max="80"
                    value={brushSize}
                    onChange={(event) => setBrushSize(Number(event.target.value))}
                  />
                </label>
                <Toggle label="Fill shapes" checked={shapeFill} onChange={setShapeFill} />
                {tool === "text" && (
                  <label className="text-field">
                    Text to place
                    <input
                      aria-label="Text to place"
                      value={text}
                      onChange={(event) => setText(event.target.value)}
                    />
                    <small>Click the canvas to place your text.</small>
                  </label>
                )}
              </PanelSection>
            </>
          ) : (
            <>
              <RecognitionPanel
                result={result}
                busy={busy}
                error={error}
                onAnalyze={() => void analyze()}
                canAnalyze={Boolean(image)}
              />
              <PanelSection title="Overlay">
                <Toggle
                  label="Show hand landmarks"
                  checked={showLandmarks}
                  onChange={setShowLandmarks}
                />
                <p className="panel-note">
                  Landmarks are a preview and are not included in the exported image.
                </p>
              </PanelSection>
            </>
          )}
        </div>
      </aside>
      {newDialog && (
        <div className="modal-backdrop">
          <form
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="New canvas"
            onSubmit={(event) => {
              event.preventDefault();
              createCanvas();
            }}
          >
            <div className="modal-heading">
              <h2>New canvas</h2>
              <IconButton label="Close new canvas" onClick={() => setNewDialog(false)}>
                <X size={18} />
              </IconButton>
            </div>
            <div className="dimension-fields">
              <label>
                Width
                <input
                  aria-label="New canvas width"
                  type="number"
                  min="1"
                  max="4096"
                  required
                  value={width}
                  onChange={(event) => setWidth(Number(event.target.value))}
                />
              </label>
              <label>
                Height
                <input
                  aria-label="New canvas height"
                  type="number"
                  min="1"
                  max="4096"
                  required
                  value={height}
                  onChange={(event) => setHeight(Number(event.target.value))}
                />
              </label>
            </div>
            <button type="submit" className="button primary full">
              <Plus size={16} />
              Create canvas
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function validDimensions(width: number, height: number) {
  return (
    Number.isInteger(width) &&
    Number.isInteger(height) &&
    width > 0 &&
    height > 0 &&
    width <= 4096 &&
    height <= 4096 &&
    width * height <= 16_777_216
  );
}
