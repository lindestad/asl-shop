import { app, BrowserWindow, dialog, ipcMain, Menu, session } from "electron";
import { readFile, writeFile, stat, mkdtemp } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { RecognitionService } from "./recognition.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const devUrl =
  !app.isPackaged && process.env.ASL_DEV_URL === "http://127.0.0.1:5173"
    ? process.env.ASL_DEV_URL
    : null;
const pageUrl = devUrl || pathToFileURL(path.join(root, "dist", "index.html")).href;
const recognition = new RecognitionService(
  app.isPackaged ? process.resourcesPath : root,
  app.isPackaged,
);
let window;

function isAppUrl(url) {
  return url.split("#")[0] === pageUrl || url.split("#")[0] === `${pageUrl}/`;
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
function decodeImage(image) {
  if (
    !image ||
    typeof image.name !== "string" ||
    typeof image.dataUrl !== "string" ||
    image.dataUrl.length > 100 * 1024 * 1024
  )
    throw new Error("Invalid image.");
  const match = /^data:image\/(png|jpeg);base64,([A-Za-z0-9+/=]+)$/.exec(image.dataUrl);
  if (!match) throw new Error("Export a PNG or JPEG image.");
  return { bytes: Buffer.from(match[2], "base64"), extension: match[1] === "jpeg" ? "jpg" : "png" };
}
function safeName(name) {
  return (
    path
      .parse(path.basename(name))
      .name.replace(/[^a-zA-Z0-9._-]/g, "_")
      .slice(0, 100) || "image"
  );
}

handle("images:open", async (multiple) => {
  const result = await dialog.showOpenDialog(window, {
    properties: multiple === true ? ["openFile", "multiSelections"] : ["openFile"],
    filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }],
  });
  if (result.canceled) return [];
  if (result.filePaths.length > 100) throw new Error("Open up to 100 images at a time.");
  const images = [];
  for (const filename of result.filePaths) {
    if ((await stat(filename)).size > 30 * 1024 * 1024)
      throw new Error(`${path.basename(filename)} is larger than 30 MB.`);
    const extension = path.extname(filename).toLowerCase();
    const mime = { ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp" }[extension];
    if (!mime) throw new Error("Unsupported image format.");
    images.push({
      name: path.basename(filename),
      dataUrl: `data:image/${mime};base64,${(await readFile(filename)).toString("base64")}`,
    });
  }
  return images;
});
handle("images:save", async (image) => {
  const { bytes, extension } = decodeImage(image);
  const result = await dialog.showSaveDialog(window, {
    defaultPath: `${safeName(image.name)}.${extension}`,
    filters: [{ name: extension.toUpperCase(), extensions: [extension] }],
  });
  if (result.canceled || !result.filePath) return null;
  await writeFile(result.filePath, bytes);
  return result.filePath;
});
handle("images:save-batch", async (images) => {
  if (!Array.isArray(images) || images.length === 0 || images.length > 100)
    throw new Error("Export between 1 and 100 images.");
  const decoded = images.map(decodeImage);
  const result = await dialog.showOpenDialog(window, {
    properties: ["openDirectory", "createDirectory"],
    title: "Choose an output folder",
  });
  if (result.canceled) return null;
  const output = await mkdtemp(path.join(result.filePaths[0], "asl-shop-"));
  const report = [];
  for (const [index, image] of images.entries()) {
    const filename = `${String(index + 1).padStart(3, "0")}-${safeName(image.name)}.${decoded[index].extension}`;
    await writeFile(path.join(output, filename), decoded[index].bytes, { flag: "wx" });
    report.push({ file: filename, recognition: image.recognition });
  }
  await writeFile(path.join(output, "recognition.json"), JSON.stringify(report, null, 2), {
    flag: "wx",
  });
  return output;
});
handle("recognition:analyze", (dataUrl) => recognition.analyze(dataUrl));
handle("recognition:frame", (dataUrl) => recognition.analyze(dataUrl, true));

async function createWindow() {
  window = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1000,
    minHeight: 720,
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
