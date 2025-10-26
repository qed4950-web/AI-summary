const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("toolbarAPI", {
  close: () => ipcRenderer.send("toolbar:close"),
  onFadeOut: (callback) => {
    ipcRenderer.on("toolbar:fade-out", () => {
      if (typeof callback === "function") {
        callback();
      }
    });
  },
  runMeeting: (payload) => ipcRenderer.invoke("run-meeting-agent", payload),
  runKnowledge: (payload) => ipcRenderer.invoke("run-knowledge-agent", payload),
  openPath: (targetPath) => ipcRenderer.invoke("open-path", targetPath),
  loadSmartFolders: () => ipcRenderer.invoke("load-smart-folders"),
  onSmartFoldersChanged: (callback) => {
    if (typeof callback !== "function") {
      return;
    }
    ipcRenderer.on("smart-folders:changed", () => callback());
  },
  pipeline: {
    status: (baseUrl) => ipcRenderer.invoke("pipeline:status", { baseUrl }),
    run: (baseUrl, payload) => ipcRenderer.invoke("pipeline:run", { baseUrl, payload }),
    cancel: (baseUrl) => ipcRenderer.invoke("pipeline:cancel", { baseUrl }),
  },
});

window.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("loaded");
});
