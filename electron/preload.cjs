const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  openImages: (multiple) => ipcRenderer.invoke("images:open", multiple),
  saveImage: (image) => ipcRenderer.invoke("images:save", image),
  saveBatch: (images) => ipcRenderer.invoke("images:save-batch", images),
  recognize: (dataUrl) => ipcRenderer.invoke("recognition:analyze", dataUrl),
  recognizeFrame: (dataUrl) => ipcRenderer.invoke("recognition:frame", dataUrl),
});
