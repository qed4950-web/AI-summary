const { app, BrowserWindow, ipcMain, screen, shell } = require("electron");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const PathToolkit = {
  baseName(targetPath) {
    if (!targetPath || typeof targetPath !== "string") {
      return "문서";
    }
    return path.basename(targetPath);
  },
  expandUser(rawPath) {
    if (!rawPath) {
      return "";
    }
    if (rawPath.startsWith("~")) {
      return path.resolve(path.join(os.homedir(), rawPath.slice(1)));
    }
    if (path.isAbsolute(rawPath)) {
      return rawPath;
    }
    return path.resolve(path.join(__dirname, "..", "..", rawPath));
  },
};

let windowRef;
const SMART_FOLDER_CONFIG = path.join(__dirname, "..", "..", "core", "config", "smart_folders.json");
const SMART_POLICY_DIR = path.join(__dirname, "..", "..", "core", "data_pipeline", "policies");

let smartFolderWatcher;
let smartPolicyWatcher;

function setupSmartFolderWatchers(targetWindow) {
  const pushUpdate = () => {
    if (targetWindow && !targetWindow.isDestroyed()) {
      targetWindow.webContents.send("smart-folders:changed");
    }
  };

  if (!smartFolderWatcher) {
    try {
      smartFolderWatcher = fs.watch(SMART_FOLDER_CONFIG, { persistent: false }, () => {
        pushUpdate();
      });
    } catch (err) {
      console.warn("[SmartFolders] watcher setup failed", err?.message || err);
    }
  }

  if (!smartPolicyWatcher) {
    try {
      smartPolicyWatcher = fs.watch(SMART_POLICY_DIR, { recursive: true, persistent: false }, () => {
        pushUpdate();
      });
    } catch (err) {
      console.warn("[SmartPolicies] watcher setup failed", err?.message || err);
    }
  }
}

function teardownSmartFolderWatchers() {
  if (smartFolderWatcher) {
    try {
      smartFolderWatcher.close();
    } catch (err) {
      console.warn("[SmartFolders] watcher close failed", err?.message || err);
    }
    smartFolderWatcher = null;
  }
  if (smartPolicyWatcher) {
    try {
      smartPolicyWatcher.close();
    } catch (err) {
      console.warn("[SmartPolicies] watcher close failed", err?.message || err);
    }
    smartPolicyWatcher = null;
  }
}

function readSmartFolders() {
  try {
    const raw = fs.readFileSync(SMART_FOLDER_CONFIG, "utf-8");
    const entries = JSON.parse(raw);
    if (!Array.isArray(entries)) {
      return [];
    }
    return entries.map((entry) => {
      const resolvedPath = PathToolkit.expandUser(entry.path || "");
      return {
        id: entry.id || entry.label,
        label: entry.label || entry.id || "폴더",
        type: entry.type || "documents",
        rawPath: entry.path || "",
        path: resolvedPath,
        scope: entry.scope || "auto",
        policyPath: entry.policy || "",
      };
    });
  } catch (err) {
    console.error("[SmartFolders] load error", err);
    return [];
  }
}

function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;
  const winWidth = 600;
  const winHeight = 420;

  const win = new BrowserWindow({
    width: winWidth,
    height: winHeight,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: true,
    hasShadow: false,
    show: false,
    webPreferences: {
      preload: `${__dirname}/preload.js`,
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const x = Math.round((width - winWidth) / 2);
  const y = Math.max(0, Math.round(height - winHeight - 48));
  win.setBounds({ x, y, width: winWidth, height: winHeight });

  win.loadFile("index.html");
  win.once("ready-to-show", () => {
    win.showInactive();
  });

  windowRef = win;
  setupSmartFolderWatchers(win);
  return win;
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("before-quit", () => {
  teardownSmartFolderWatchers();
});

ipcMain.on("toolbar:close", () => {
  if (!windowRef || windowRef.isDestroyed()) {
    return;
  }
  windowRef.webContents.send("toolbar:fade-out");
  setTimeout(() => {
    if (windowRef && !windowRef.isDestroyed()) {
      windowRef.destroy();
    }
  }, 220);
});

ipcMain.handle("run-meeting-agent", async (_event, payload) => {
  const { query, folder } = payload || {};
  const folderContext = folder || {
    label: "선택된 폴더",
    path: "",
    scope: "local",
    policyPath: "",
  };

  return new Promise((resolve) => {
    try {
      const scriptPath = path.join(__dirname, "..", "..", "scripts", "run_meeting_agent.py");
      const args = [
        scriptPath,
        "--folder-path",
        folderContext.path || "",
        "--folder-label",
        folderContext.label || "",
        "--folder-scope",
        folderContext.scope || "",
        "--policy-path",
        folderContext.policyPath || "",
        "--query",
        query || "",
        "--output-json",
      ];

      const child = spawn("python3", args, {
        cwd: path.join(__dirname, "..", ".."),
        shell: false,
      });

      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      const finalize = (exitCode) => {
        let parsed = null;
        try {
          parsed = JSON.parse(stdout.trim() || "{}");
        } catch (err) {
          parsed = {
            ok: false,
            error: err?.message || "meeting agent 응답을 처리하지 못했습니다.",
            data: null,
          };
        }

        const response = {
          ok: Boolean(parsed?.ok) && exitCode === 0,
          data: parsed?.data || null,
          error: parsed?.error || (exitCode === 0 ? null : `meeting agent 종료 코드 ${exitCode}`),
          stderr: stderr.trim() || null,
          exitCode,
          fallback: false,
        };
        resolve(response);
      };

      child.on("close", (code) => {
        finalize(typeof code === "number" ? code : -1);
      });

      child.on("error", (err) => {
        resolve({
          ok: false,
          data: null,
          error: err.message,
          stderr: stderr.trim() || null,
          exitCode: -1,
          fallback: false,
        });
      });
    } catch (err) {
      resolve({
        ok: false,
        data: null,
        error: err.message,
        stderr: null,
        exitCode: -1,
        fallback: false,
      });
    }
  });
});

ipcMain.handle("run-knowledge-agent", async (_event, payload) => {
  const { query, folder } = payload || {};
  const folderContext = folder || {
    label: "선택된 폴더",
    path: "",
    scope: "local",
    policyPath: "",
  };

  if (!query || !query.trim()) {
    return {
      ok: false,
      data: null,
      error: "검색할 내용을 입력해주세요.",
      stderr: null,
      exitCode: -1,
      fallback: false,
    };
  }

  return new Promise((resolve) => {
    try {
      const scriptPath = path.join(__dirname, "..", "..", "scripts", "run_knowledge_agent.py");
      const args = [
        scriptPath,
        "--query",
        query,
        "--folder-path",
        folderContext.path || "",
        "--folder-label",
        folderContext.label || "",
        "--folder-scope",
        folderContext.scope || "auto",
        "--policy-path",
        folderContext.policyPath || "",
      ];

      const child = spawn("python3", args, {
        cwd: path.join(__dirname, "..", ".."),
        shell: false,
      });

      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      const finalize = (exitCode) => {
        let parsed;
        try {
          parsed = JSON.parse(stdout.trim() || "{}");
        } catch (err) {
          parsed = {
            ok: false,
            error: err?.message || "knowledge agent 응답을 처리하지 못했습니다.",
            data: null,
          };
        }

        resolve({
          ok: Boolean(parsed?.ok) && exitCode === 0,
          data: parsed?.data || null,
          error: parsed?.error || (exitCode === 0 ? null : `knowledge agent 종료 코드 ${exitCode}`),
          stderr: stderr.trim() || null,
          exitCode,
          fallback: false,
        });
      };

      child.on("close", (code) => {
        finalize(typeof code === "number" ? code : -1);
      });

      child.on("error", (err) => {
        resolve({
          ok: false,
          data: null,
          error: err.message,
          stderr: stderr.trim() || null,
          exitCode: -1,
          fallback: false,
        });
      });
    } catch (err) {
      resolve({
        ok: false,
        data: null,
        error: err.message,
        stderr: null,
        exitCode: -1,
        fallback: false,
      });
    }
  });
});

ipcMain.handle("open-path", async (_event, targetPath) => {
  if (!targetPath) {
    return { ok: false, error: "path is empty" };
  }
  try {
    const normalizedPath = path.isAbsolute(targetPath)
      ? targetPath
      : path.join(__dirname, "..", "..", targetPath);
    const result = await shell.openPath(normalizedPath);
    return { ok: result === "", error: result || null };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle("load-smart-folders", async () => {
  const data = readSmartFolders();
  return { ok: true, data };
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    teardownSmartFolderWatchers();
    app.quit();
  }
});
