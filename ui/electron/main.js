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

  return new Promise((resolve) => {
    const fallbackSummary = {
      summary: "회의 요약 결과를 생성했습니다 (모의 데이터).",
      actions: ["액션 아이템 A", "액션 아이템 B"],
      folder,
      query,
    };

    try {
      const mockScript = path.join(__dirname, "..", "..", "scripts", "pipeline", "mock_meeting_summary.py");
      const child = spawn("python3", [mockScript, query, "--folder", folder?.path ?? "", "--json"], {
        cwd: path.join(__dirname, "..", ".."),
        shell: false,
      });

      let stdout = "";
      child.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      child.on("close", () => {
        try {
          const parsed = JSON.parse(stdout.trim() || "{}");
          const normalizedFolder =
            parsed && typeof parsed.folder === "object" && parsed.folder !== null
              ? parsed.folder
              : {
                  label:
                    folder?.label ||
                    (typeof parsed.folder === "string" && parsed.folder.trim()
                      ? parsed.folder.trim()
                      : "선택된 폴더"),
                  path:
                    folder?.path ||
                    (typeof parsed.folder === "string" ? parsed.folder : ""),
                  scope: folder?.scope || "auto",
                  policyPath: folder?.policyPath || "",
                };
          const normalized = {
            ...parsed,
            folder: normalizedFolder,
            query: parsed.query || query,
            highlights: Array.isArray(parsed.highlights) ? parsed.highlights : [],
            actions: Array.isArray(parsed.actions) ? parsed.actions : [],
          };
          resolve({ ok: true, data: normalized, fallback: false });
        } catch (err) {
          resolve({ ok: true, data: fallbackSummary, fallback: true, error: err?.message });
        }
      });

      child.on("error", (err) => {
        resolve({ ok: false, error: err.message, data: fallbackSummary, fallback: true });
      });
    } catch (err) {
      resolve({ ok: false, error: err.message, data: fallbackSummary, fallback: true });
    }
  });
});

ipcMain.handle("run-knowledge-agent", async (_event, payload) => {
  const { query, folder } = payload || {};

  return new Promise((resolve) => {
    const scriptPath = path.join(__dirname, "..", "..", "scripts", "pipeline", "infopilot.py");
    const baseArgs = [
      "chat",
      "--model",
      "data/topic_model.joblib",
      "--corpus",
      "data/corpus.parquet",
      "--cache",
      "data/cache",
    ];
    if (folder?.scope && folder.scope !== "auto") {
      baseArgs.push("--scope", folder.scope);
    }
    if (folder?.policyPath) {
      baseArgs.push("--policy", folder.policyPath);
    }
    const args = [...baseArgs, "--query", query, "--json"];

    const child = spawn("python3", [scriptPath, ...args], {
      cwd: path.join(__dirname, "..", ".."),
      shell: false,
    });

    let stdout = "";
    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    const handleMockFallback = (errorMessage) => {
      const mockScript = path.join(__dirname, "..", "..", "scripts", "pipeline", "mock_knowledge_search.py");
      const mockChild = spawn("python3", [mockScript, query, "--folder", folder?.path ?? "", "--json"], {
        cwd: path.join(__dirname, "..", ".."),
        shell: false,
      });
      let mockStdout = "";
      mockChild.stdout.on("data", (d) => {
        mockStdout += d.toString();
      });
      mockChild.on("close", () => {
        try {
          const parsed = JSON.parse(mockStdout.trim());
          resolve({
            ok: true,
            fallback: true,
            error: errorMessage,
            data: {
              query,
              folder,
              items: Array.isArray(parsed.items)
                ? parsed.items.map((hit) => ({
                    title: hit.title,
                    snippet: hit.snippet,
                    path: hit.path,
                    score: null,
                  }))
                : [],
            },
          });
        } catch (err) {
          resolve({ ok: false, data: null, fallback: true, error: err?.message || errorMessage });
        }
      });
      mockChild.on("error", (err) => {
        resolve({ ok: false, data: null, fallback: true, error: err.message || errorMessage });
      });
    };

    child.on("close", (code) => {
      if (code !== 0) {
        handleMockFallback(`infopilot chat exited with code ${code}`);
        return;
      }
      try {
        const parsed = JSON.parse(stdout.trim());
        const normalized = {
          query: parsed.query,
          folder,
          items: Array.isArray(parsed.results)
            ? parsed.results.map((hit) => ({
                title: hit.title || PathToolkit.baseName(hit.path),
                snippet: hit.preview || hit.snippet || "",
                path: hit.path,
                score: hit.score,
              }))
            : [],
        };
        resolve({ ok: true, data: normalized, fallback: false });
      } catch (err) {
        handleMockFallback(err?.message || "JSON parse error");
      }
    });

    child.on("error", (err) => {
      handleMockFallback(err.message);
    });
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
    app.quit();
  }
});
