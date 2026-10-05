import { useRef, useState } from "react";
import {
  ArrowUpRight,
  Camera,
  ChevronRight,
  Command,
  Files,
  Hand,
  Image,
  Plus,
  PanelLeftClose,
  X,
} from "lucide-react";
import Editor from "./Editor";
import CameraWorkspace from "./Camera";
import Batch from "./Batch";
import { readFiles } from "./image";
import type { SourceImage, Workspace } from "./types";
import { IconButton } from "./components";

const workspaceInfo = {
  editor: { title: "Image editor", icon: Image },
  camera: { title: "Live camera", icon: Camera },
  batch: { title: "Batch studio", icon: Files },
};

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>("editor");
  const [incoming, setIncoming] = useState<SourceImage | null>(null);
  const [batchImages, setBatchImages] = useState<SourceImage[]>([]);
  const [newRequest, setNewRequest] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [help, setHelp] = useState(false);
  const [compact, setCompact] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const target = useRef<Workspace>("editor");
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const info = workspaceInfo[workspace];
  function notify(message: string) {
    setNotice(message);
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), 6000);
  }
  function acceptImages(images: SourceImage[], destination: Workspace) {
    if (!images.length) return;
    if (destination === "batch") setBatchImages(images);
    else setIncoming(images[0]);
    setWorkspace(destination === "camera" ? "editor" : destination);
  }
  async function openImages(destination: Workspace = workspace) {
    target.current = destination;
    if (!window.desktop) {
      fileInput.current?.click();
      return;
    }
    try {
      acceptImages(await window.desktop.openImages(destination === "batch"), destination);
    } catch (error) {
      notify((error as Error).message);
    }
  }
  return (
    <div className={`app ${compact ? "compact-nav" : ""}`}>
      <aside className="sidebar">
        <button className="brand" onClick={() => setWorkspace("editor")} aria-label="ASL Shop home">
          <span className="brand-mark">
            <Hand size={23} strokeWidth={1.5} />
          </span>
          <span className="brand-wordmark">
            asl<span>shop</span>
          </span>
        </button>
        <nav className="workspace-nav" aria-label="Workspaces">
          {(Object.keys(workspaceInfo) as Workspace[]).map((key) => {
            const Icon = workspaceInfo[key].icon;
            return (
              <button
                key={key}
                className={workspace === key ? "active" : ""}
                onClick={() => setWorkspace(key)}
                title={workspaceInfo[key].title}
                aria-label={workspaceInfo[key].title}
              >
                <Icon size={18} strokeWidth={1.5} />
                <span>{workspaceInfo[key].title}</span>
                {workspace === key && <ChevronRight size={14} className="nav-chevron" />}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-divider" />
        <button
          className="sidebar-action"
          aria-label="New canvas"
          onClick={() => {
            setWorkspace("editor");
            setNewRequest((value) => value + 1);
          }}
        >
          <Plus size={17} />
          <span>New canvas</span>
        </button>
        <button
          className="sidebar-action"
          aria-label="Open image"
          onClick={() => void openImages("editor")}
        >
          <ArrowUpRight size={17} />
          <span>Open image</span>
        </button>
        <div className="sidebar-bottom">
          <button className="shortcut-button" aria-label="Help" onClick={() => setHelp(true)}>
            <Command size={15} />
            <span>Help</span>
          </button>
          <div className="sidebar-footer">
            <span>v0.1</span>
            <IconButton
              label={compact ? "Expand sidebar" : "Collapse sidebar"}
              onClick={() => setCompact(!compact)}
            >
              <PanelLeftClose size={15} />
            </IconButton>
          </div>
        </div>
      </aside>
      <main className="main-shell">
        <header className="app-header">
          <h1>{info.title}</h1>
          <div className="header-right">
            {!window.desktop && <span className="runtime-badge">Browser preview</span>}
            <button
              className="button primary"
              onClick={() => void openImages(workspace === "batch" ? "batch" : "editor")}
            >
              <Plus size={16} />
              {workspace === "batch" ? "Add images" : "Open image"}
            </button>
          </div>
        </header>
        <Editor
          active={workspace === "editor"}
          incoming={incoming}
          newRequest={newRequest}
          onOpen={() => void openImages("editor")}
          onCamera={() => setWorkspace("camera")}
          notify={notify}
        />
        <CameraWorkspace
          active={workspace === "camera"}
          onCapture={(image) => acceptImages([image], "editor")}
          notify={notify}
        />
        <Batch
          active={workspace === "batch"}
          incoming={batchImages}
          onOpen={() => void openImages("batch")}
          notify={notify}
        />
      </main>
      <input
        ref={fileInput}
        data-testid="image-input"
        className="visually-hidden"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        multiple={workspace === "batch"}
        onChange={async (event) => {
          try {
            acceptImages(await readFiles(Array.from(event.target.files ?? [])), target.current);
          } catch (error) {
            notify((error as Error).message);
          }
          event.target.value = "";
        }}
      />
      {notice && (
        <div className="toast" role="status">
          <span>{notice}</span>
          <IconButton label="Dismiss notification" onClick={() => setNotice(null)}>
            <X size={14} />
          </IconButton>
        </div>
      )}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="modal guide-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Help"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-heading">
              <h2>Help</h2>
              <IconButton label="Close help" onClick={() => setHelp(false)}>
                <X size={18} />
              </IconButton>
            </div>
            <dl className="guide-list">
              <dt>Image editor</dt>
              <dd>
                Draw, crop, transform, filter, and export PNGs. Ctrl/⌘ Z: undo. Ctrl/⌘ Shift Z:
                redo. Ctrl/⌘ S: export.
              </dd>
              <dt>Live camera</dt>
              <dd>Detect hand landmarks and capture a frame for the editor.</dd>
              <dt>Batch studio</dt>
              <dd>Analyze images and export PNGs with a JSON recognition report.</dd>
              <dt>Letter recognition</dt>
              <dd>Hand detection is available. Letter classification requires a trained model.</dd>
            </dl>
            <button className="button primary full" onClick={() => setHelp(false)}>
              Close
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
