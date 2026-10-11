const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  openImage: () => ipcRenderer.invoke("images:open"),
  recognize: (dataUrl) => ipcRenderer.invoke("recognition:analyze", dataUrl),
  recognizeFrame: (dataUrl) => ipcRenderer.invoke("recognition:frame", dataUrl),
});
