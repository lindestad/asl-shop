import { app, BrowserWindow, dialog, ipcMain, Menu, net, protocol, session } from "electron";
import { readFile, stat } from "node:fs/promises";
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
    throw new Error("Bildet må være mindre enn 16 MB.");
  const mime = { ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp" }[
    path.extname(filename).toLowerCase()
  ];
  if (!mime) throw new Error("Ugyldig bildeformat.");
  return {
    name: path.basename(filename),
    dataUrl: `data:image/${mime};base64,${(await readFile(filename)).toString("base64")}`,
  };
});
handle("recognition:analyze", (dataUrl) => recognition.analyze(dataUrl));
handle("recognition:frame", (dataUrl) => recognition.analyze(dataUrl, true));

async function createWindow() {
  window = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 760,
    minHeight: 600,
    title: "ASL Shop",
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
