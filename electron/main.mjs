import { app, BrowserWindow, dialog, ipcMain, Menu, net, protocol, session } from "electron";
import { readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { Readable } from "node:stream";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { RecognitionService } from "./recognition.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const devUrl =
  !app.isPackaged && process.env.ASL_DEV_URL === "http://127.0.0.1:5173"
    ? process.env.ASL_DEV_URL
    : null;
const pageUrl = devUrl || "app://bundle/index.html";
protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true } },
  {
    scheme: "demo-media",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      stream: true,
      corsEnabled: true,
    },
  },
]);
const recognition = new RecognitionService(
  app.isPackaged ? process.resourcesPath : root,
  app.isPackaged,
);
let window;

function isAppUrl(url) {
  return url.split("#")[0] === pageUrl || url.split("#")[0] === `${pageUrl}/`;
}
function registerAppProtocol() {
  const dist = path.join(root, "dist");
  protocol.handle("app", (request) => {
    const url = new URL(request.url);
    if (url.host !== "bundle" || request.method !== "GET")
      return new Response("Not found", { status: 404 });
    const pathname = decodeURIComponent(url.pathname);
    const filename = path.resolve(dist, `.${pathname === "/" ? "/index.html" : pathname}`);
    const relative = path.relative(dist, filename);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
      return new Response("Not found", { status: 404 });
    return net.fetch(pathToFileURL(filename).href);
  });
}
// Demo replay footage: demo-media://clips/<file>.mp4 serves only files from the
// dataset's demo-clips directory; nothing else on disk is reachable. Media
// playback needs real Range/206 support, which net.fetch does not provide.
async function serveDemoClip(request) {
  const url = new URL(request.url);
  if (url.host !== "clips" || (request.method !== "GET" && request.method !== "HEAD"))
    return new Response("Not found", { status: 404 });
  const name = path.basename(decodeURIComponent(url.pathname));
  if (!name.endsWith(".mp4")) return new Response("Not found", { status: 404 });
  const file = path.join(root, "data", "external", "openfs", "demo-clips", name);
  let info;
  try {
    info = await stat(file);
  } catch {
    return new Response("Not found", { status: 404 });
  }
  // CORS headers keep the drawn canvas untainted so frames can be exported to
  // the recognition worker (toDataURL rejects tainted canvases).
  const headers = {
    "Content-Type": "video/mp4",
    "Accept-Ranges": "bytes",
    "Access-Control-Allow-Origin": "*",
  };
  let start = 0;
  let end = info.size - 1;
  let partial = false;
  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get("range") || "");
  if (range) {
    partial = true;
    if (range[1]) {
      start = Number(range[1]);
      if (range[2]) end = Math.min(Number(range[2]), info.size - 1);
    } else if (range[2]) {
      start = Math.max(0, info.size - Number(range[2]));
    }
    if (start > end || start >= info.size)
      return new Response(null, {
        status: 416,
        headers: { "Content-Range": `bytes */${info.size}` },
      });
  }
  headers["Content-Length"] = String(end - start + 1);
  if (partial) headers["Content-Range"] = `bytes ${start}-${end}/${info.size}`;
  if (request.method === "HEAD")
    return new Response(null, { status: partial ? 206 : 200, headers });
  return new Response(Readable.toWeb(createReadStream(file, { start, end })), {
    status: partial ? 206 : 200,
    headers,
  });
}
function handle(channel, callback) {
  ipcMain.handle(channel, (event, ...args) => {
    if (
      event.sender !== window?.webContents ||
      event.senderFrame !== window.webContents.mainFrame ||
      !isAppUrl(event.senderFrame.url)
    )
      throw new Error("Unknown application window.");
    return callback(...args);
  });
}
handle("images:open", async () => {
  const result = await dialog.showOpenDialog(window, {
    properties: ["openFile"],
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }],
  });
  if (result.canceled || !result.filePaths[0]) return null;
  const filename = result.filePaths[0];
  if ((await stat(filename)).size > 16 * 1024 * 1024)
    throw new Error("The image must be smaller than 16 MB.");
  const mime = { ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp" }[
    path.extname(filename).toLowerCase()
  ];
  if (!mime) throw new Error("Unsupported image format.");
  return {
    name: path.basename(filename),
    dataUrl: `data:image/${mime};base64,${(await readFile(filename)).toString("base64")}`,
  };
});
handle("recognition:analyze", (dataUrl, model) => recognition.analyze(dataUrl, false, { model }));
handle("recognition:frame", (dataUrl, options) => recognition.analyze(dataUrl, true, options));
handle("recognition:landmarks", (landmarks, options) =>
  recognition.analyzeLandmarks(landmarks, options),
);
handle("demo:list", () => recognition.demoList());
handle("demo:clip", (index) => recognition.demoClip(index));

async function createWindow() {
  window = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 760,
    minHeight: 600,
    title: "ASL alphabet recognition",
    icon: path.join(root, "assets", "icon.png"),
    backgroundColor: "#fafafa",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(root, "electron", "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event, url) => {
    if (!isAppUrl(url)) event.preventDefault();
  });
  window.on("closed", () => {
    recognition.stop();
    window = null;
  });
  await window.loadURL(pageUrl);
}

app
  .whenReady()
  .then(async () => {
    if (!devUrl) registerAppProtocol();
    protocol.handle("demo-media", serveDemoClip);
    Menu.setApplicationMenu(
      Menu.buildFromTemplate([
        ...(process.platform === "darwin" ? [{ role: "appMenu" }] : []),
        { role: "editMenu" },
        { role: "viewMenu" },
        { role: "windowMenu" },
      ]),
    );
    session.defaultSession.setPermissionCheckHandler(
      (contents, permission) =>
        contents === window?.webContents && permission === "media" && isAppUrl(contents.getURL()),
    );
    session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) =>
      callback(
        contents === window?.webContents &&
          permission === "media" &&
          isAppUrl(details.requestingUrl) &&
          details.mediaTypes?.every((type) => type === "video"),
      ),
    );
    await createWindow();
  })
  .catch((error) => {
    console.error(error);
    app.quit();
  });
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) void createWindow();
});
app.on("window-all-closed", () => {
  recognition.stop();
  if (process.platform !== "darwin") app.quit();
});
app.on("before-quit", () => recognition.stop());
