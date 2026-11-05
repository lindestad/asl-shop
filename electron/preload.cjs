const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  openImage: () => ipcRenderer.invoke("images:open"),
  recognize: (dataUrl, model) => ipcRenderer.invoke("recognition:analyze", dataUrl, model),
  recognizeFrame: (dataUrl, model) => ipcRenderer.invoke("recognition:frame", dataUrl, model),
});
