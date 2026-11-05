const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  openImage: () => ipcRenderer.invoke("images:open"),
  recognize: (dataUrl, model) => ipcRenderer.invoke("recognition:analyze", dataUrl, model),
  recognizeFrame: (dataUrl, options) => ipcRenderer.invoke("recognition:frame", dataUrl, options),
  recognizeLandmarks: (landmarks, options) =>
    ipcRenderer.invoke("recognition:landmarks", landmarks, options),
  demoList: () => ipcRenderer.invoke("demo:list"),
  demoClip: (index) => ipcRenderer.invoke("demo:clip", index),
});
