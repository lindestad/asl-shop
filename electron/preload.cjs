const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  openImage: () => ipcRenderer.invoke("images:open"),
  recognize: (dataUrl) => ipcRenderer.invoke("recognition:analyze", dataUrl),
  recognizeFrame: (dataUrl, options) => ipcRenderer.invoke("recognition:frame", dataUrl, options),
  recognizeLandmarks: (landmarks, options) =>
    ipcRenderer.invoke("recognition:landmarks", landmarks, options),
  demoList: () => ipcRenderer.invoke("demo:list"),
  demoClip: (index) => ipcRenderer.invoke("demo:clip", index),
});
