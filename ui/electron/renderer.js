(() => {
  const toolbar = document.querySelector(".toolbar");
  const closeBtn = document.querySelector(".toolbar__right .icon.close");
  const status = toolbar?.querySelector(".status");
  const actionButtons = toolbar?.querySelectorAll(".icon[data-action]") ?? [];
  const commandForm = document.getElementById("command-form");
  const commandInput = document.getElementById("command-input");
  const smartFoldersSheet = document.getElementById("smart-folders");
  const smartFolderList = document.getElementById("smart-folder-list");
  const smartFolderClose = smartFoldersSheet?.querySelector('[data-action="close-smart-folders"]');
  const meetingSummaryPanel = document.getElementById("meeting-summary");
  const meetingSummaryFolder = meetingSummaryPanel?.querySelector(".meeting-summary__folder");
  const meetingSummaryText = meetingSummaryPanel?.querySelector(".meeting-summary__text");
  const meetingSummaryActions = meetingSummaryPanel?.querySelector(".meeting-summary__actions");
  const meetingSummaryHighlights = meetingSummaryPanel?.querySelector(".meeting-summary__highlights");
  const meetingSummaryClose = meetingSummaryPanel?.querySelector('[data-action="close-meeting-summary"]');
  const knowledgeResultsPanel = document.getElementById("knowledge-results");
  const knowledgeResultsFolder = knowledgeResultsPanel?.querySelector(".knowledge-results__folder");
  const knowledgeResultsList = knowledgeResultsPanel?.querySelector(".knowledge-results__list");
  const knowledgeResultsClose = knowledgeResultsPanel?.querySelector('[data-action="close-knowledge-results"]');

  if (!toolbar || !closeBtn || !commandForm || !commandInput) {
    return;
  }

  let isClosing = false;
  let statusTimer = null;
  let micActive = false;
  let smartFolders = [];
  let selectedFolder = null;

  const updateCommandPlaceholder = () => {
    if (!commandInput) {
      return;
    }
    if (!selectedFolder || (selectedFolder.scope || "auto") === "global") {
      commandInput.placeholder = "무엇을 도와드릴까요? 예: 회의록 요약해줘";
    } else {
      commandInput.placeholder = `${selectedFolder.label} 범위에서 명령을 입력하세요.`;
    }
  };

  const baseName = (targetPath = "") => {
    if (!targetPath) {
      return "문서";
    }
    const normalized = targetPath.replace(/\\/g, "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || targetPath;
  };

  const showStatus = (message) => {
    if (!status) {
      return;
    }
    status.textContent = message;
    status.classList.add("show");
    if (statusTimer) {
      clearTimeout(statusTimer);
    }
    statusTimer = setTimeout(() => {
      status.classList.remove("show");
    }, 2200);
  };

  const fadeOut = () => {
    if (isClosing) {
      return;
    }
    isClosing = true;
    toolbar.classList.add("fade-out");
    setTimeout(() => {
      window.close();
    }, 220);
  };

  const toggleSmartFolders = (show) => {
    if (!smartFoldersSheet) {
      return;
    }
    if (typeof show === "boolean") {
      smartFoldersSheet.hidden = !show;
    } else {
      smartFoldersSheet.hidden = !smartFoldersSheet.hidden;
    }
    if (!smartFoldersSheet.hidden && smartFolderList && selectedFolder) {
      Array.from(smartFolderList.children).forEach((node) => {
        const matches =
          node.dataset.id === selectedFolder.id &&
          (node.dataset.path || "") === (selectedFolder.path || "");
        node.classList.toggle("is-active", matches);
      });
    }
  };

  const closePanels = () => {
    if (meetingSummaryPanel) meetingSummaryPanel.hidden = true;
    if (knowledgeResultsPanel) knowledgeResultsPanel.hidden = true;
  };

  const highlightSmartFolders = () => {
    if (!smartFolderList || !selectedFolder) {
      return;
    }
    Array.from(smartFolderList.children).forEach((node) => {
      const matches =
        node.dataset.id === selectedFolder.id &&
        (node.dataset.path || "") === (selectedFolder.rawPath || selectedFolder.path || "");
      node.classList.toggle("is-active", matches);
    });
  };

  const setSelectedFolder = (folder) => {
    selectedFolder = folder;
    updateCommandPlaceholder();
    highlightSmartFolders();
  };

  const renderSmartFolders = (folders) => {
    if (!smartFolderList) {
      return;
    }
    smartFolderList.innerHTML = "";
    folders.forEach((folder) => {
      const li = document.createElement("li");
      li.dataset.id = folder.id || folder.label;
      li.dataset.folder = folder.type || "documents";
      li.dataset.path = folder.rawPath || folder.path || "";
      li.dataset.scope = folder.scope || "auto";
      li.dataset.policy = folder.policyPath || "";
      li.textContent = `${folder.icon || "📄"} ${folder.label}`;
      smartFolderList.appendChild(li);
    });
    highlightSmartFolders();
  };

  const runMeetingAgent = async (query) => {
    closePanels();
    showStatus("회의 요약 준비 중…");
    if (window.toolbarAPI && typeof window.toolbarAPI.runMeeting === "function") {
      try {
        const folderContext =
          selectedFolder || {
            label: "전체 검색",
            type: "global",
            path: "",
            rawPath: "",
            scope: "global",
            policyPath: "",
          };
        const response = await window.toolbarAPI.runMeeting({ query, folder: folderContext });
        console.log("[MeetingAgent]", response);
        if (response?.ok) {
          showStatus("회의 요약 결과가 생성되었습니다.");
          toolbar.dispatchEvent(new CustomEvent("meeting-summary", { detail: response }));
        } else {
          showStatus("회의 요약 중 오류가 발생했습니다.");
        }
      } catch (err) {
        console.error("[MeetingAgent] IPC error", err);
        showStatus("회의 요약 중 오류가 발생했습니다.");
      }
    }
  };

  const runKnowledgeAgent = async (query) => {
    closePanels();
    showStatus("지식 검색 중…");
    if (window.toolbarAPI && typeof window.toolbarAPI.runKnowledge === "function") {
      try {
        const folderContext =
          selectedFolder || {
            label: "전체 검색",
            type: "global",
            path: "",
            rawPath: "",
            scope: "global",
            policyPath: "",
          };
        const response = await window.toolbarAPI.runKnowledge({ query, folder: folderContext });
        console.log("[KnowledgeAgent]", response);
        if (response?.ok) {
          showStatus("관련 문서 목록을 준비했습니다.");
          toolbar.dispatchEvent(new CustomEvent("knowledge-results", { detail: response }));
        } else {
          showStatus("지식 검색 중 오류가 발생했습니다.");
        }
      } catch (err) {
        console.error("[KnowledgeAgent] IPC error", err);
        showStatus("지식 검색 중 오류가 발생했습니다.");
      }
    }
  };

  const detectIntent = (text) => {
    const normalized = text.toLowerCase();
    const meetingTriggers = ["회의", "회의록", "요약", "음성", "transcript"];
    const knowledgeTriggers = ["찾아", "어딨어", "보고해", "검색", "summary"]; // English fallback

    if (meetingTriggers.some((word) => normalized.includes(word))) {
      return "meeting";
    }
    if (knowledgeTriggers.some((word) => normalized.includes(word))) {
      return "knowledge";
    }
    return null;
  };

  closeBtn.addEventListener("click", () => {
    fadeOut();
    if (window.toolbarAPI && typeof window.toolbarAPI.close === "function") {
      window.toolbarAPI.close();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (smartFoldersSheet && !smartFoldersSheet.hidden) {
        toggleSmartFolders(false);
      } else {
        closePanels();
        closeBtn.click();
      }
    }
  });

  if (window.toolbarAPI && typeof window.toolbarAPI.onFadeOut === "function") {
    window.toolbarAPI.onFadeOut(fadeOut);
  }

  actionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      switch (action) {
        case "mic": {
          micActive = !micActive;
          button.classList.toggle("is-active", micActive);
          showStatus(micActive ? "음성 명령 대기 중" : "음성 캡처 일시중지");
          break;
        }
        case "copy": {
          const fallbackText = "샘플 요약을 클립보드에 복사했습니다.";
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard
              .writeText("This is a placeholder summary snippet.")
              .then(
                () => showStatus("요약을 클립보드로 복사했습니다."),
                () => showStatus(fallbackText)
              );
          } else {
            showStatus(fallbackText);
          }
          break;
        }
        case "settings": {
          showStatus("환경 설정 패널을 준비 중입니다.");
          break;
        }
        case "smart-folders": {
          toggleSmartFolders();
          break;
        }
        default:
          break;
      }
    });
  });

  smartFolderList?.addEventListener("click", (event) => {
    const item = event.target.closest("li");
    if (!item) {
      return;
    }
    const folder =
      smartFolders.find((entry) => entry.id === item.dataset.id) ||
      {
        id: item.dataset.id || item.textContent?.trim() || "folder",
        label: item.textContent?.trim() ?? "선택된 폴더",
        type: item.dataset.folder || "global",
        path: item.dataset.path || "",
        rawPath: item.dataset.path || "",
        scope: item.dataset.scope || "auto",
        policyPath: item.dataset.policy || "",
      };
    setSelectedFolder(folder);
    const isGlobal = (folder.scope || "auto") === "global";
    showStatus(isGlobal ? "전체 검색 모드를 선택했습니다." : `${folder.label} 폴더를 선택했습니다.`);
    toggleSmartFolders(false);
  });

  smartFolderClose?.addEventListener("click", () => toggleSmartFolders(false));
  meetingSummaryClose?.addEventListener("click", () => {
    if (meetingSummaryPanel) {
      meetingSummaryPanel.hidden = true;
    }
  });
  knowledgeResultsClose?.addEventListener("click", () => {
    if (knowledgeResultsPanel) {
      knowledgeResultsPanel.hidden = true;
    }
  });

  toolbar.addEventListener("meeting-summary", (event) => {
    if (!meetingSummaryPanel || !meetingSummaryText || !meetingSummaryActions || !meetingSummaryFolder || !meetingSummaryHighlights) {
      return;
    }
    const detail = event.detail || {};
    const { data, fallback } = detail;
    const summaryText = data?.summary || "요약 결과를 불러오지 못했습니다.";
    const actionItems = Array.isArray(data?.actions) ? data.actions : [];
    const highlights = Array.isArray(data?.highlights) ? data.highlights : [];
    const folderLabel = data?.folder?.label || selectedFolder.label;

    meetingSummaryFolder.textContent = fallback ? `${folderLabel} (모의 데이터)` : folderLabel;
    meetingSummaryText.textContent = summaryText;
    meetingSummaryHighlights.innerHTML = "";
    meetingSummaryActions.innerHTML = "";
    if (highlights.length) {
      highlights.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        meetingSummaryHighlights.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "강조할 하이라이트가 없습니다.";
      meetingSummaryHighlights.appendChild(li);
    }

    if (actionItems.length) {
      actionItems.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        meetingSummaryActions.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "등록된 액션 아이템이 없습니다.";
      meetingSummaryActions.appendChild(li);
    }

    meetingSummaryPanel.hidden = false;
  });

  toolbar.addEventListener("knowledge-results", (event) => {
    if (!knowledgeResultsPanel || !knowledgeResultsList || !knowledgeResultsFolder) {
      return;
    }
    const detail = event.detail || {};
    const { data, fallback } = detail;
    const items = Array.isArray(data?.items) ? data.items : [];
    const folderLabel = data?.folder?.label || selectedFolder.label;

    knowledgeResultsFolder.textContent = fallback ? `${folderLabel} (모의 데이터)` : folderLabel;
    knowledgeResultsList.innerHTML = "";

    if (items.length === 0) {
      const li = document.createElement("li");
      li.textContent = "관련 문서를 찾지 못했습니다.";
      knowledgeResultsList.appendChild(li);
    } else {
      items.forEach((item) => {
        const li = document.createElement("li");
        const title = document.createElement("p");
        title.className = "knowledge-results__title";
        title.textContent = item.title || baseName(item.path);

        const snippet = document.createElement("p");
        snippet.className = "knowledge-results__snippet";
        snippet.textContent = item.snippet || "미리보기가 없습니다.";

        const path = document.createElement("p");
        path.className = "knowledge-results__snippet";
        path.textContent = item.path || "경로 정보 없음";

        const actions = document.createElement("div");
        actions.className = "knowledge-results__actions";
        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.dataset.action = "open";
        openBtn.dataset.path = item.path || "";
        openBtn.dataset.title = item.title || "";
        openBtn.textContent = "열기";

        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.dataset.action = "copy";
        copyBtn.dataset.path = item.path || "";
        copyBtn.dataset.title = item.title || "";
        copyBtn.textContent = "경로 복사";

        actions.appendChild(openBtn);
        actions.appendChild(copyBtn);

        li.appendChild(title);
        li.appendChild(snippet);
        li.appendChild(path);
        li.appendChild(actions);
        knowledgeResultsList.appendChild(li);
      });
    }

    knowledgeResultsPanel.hidden = false;
  });

  const loadSmartFolders = async () => {
    if (!window.toolbarAPI || typeof window.toolbarAPI.loadSmartFolders !== "function") {
      smartFolders = [
        {
          id: "global",
          label: "🌐 전체 검색 (모든 문서)",
          type: "global",
          path: "",
          rawPath: "",
          scope: "global",
          policyPath: "",
          icon: "🌐",
        },
      ];
      renderSmartFolders(smartFolders);
      setSelectedFolder(smartFolders[0]);
      return;
    }
    try {
      const result = await window.toolbarAPI.loadSmartFolders();
      const data = Array.isArray(result?.data) ? result.data : [];
      if (data.length === 0) {
        throw new Error("no smart folders");
      }
      smartFolders = data.map((entry) => ({
        id: entry.id || entry.label,
        label: entry.label || entry.id || "폴더",
        type: entry.type || "documents",
        path: entry.path || "",
        rawPath: entry.rawPath || "",
        scope: entry.scope || "auto",
        policyPath: entry.policyPath || "",
        icon: entry.type === "meeting" ? "📅" : entry.type === "hr" ? "👤" : entry.type === "global" ? "🌐" : "📄",
      }));
    } catch (err) {
      console.error("[SmartFolders] load error", err);
      smartFolders = [
        {
          id: "global",
          label: "🌐 전체 검색 (모든 문서)",
          type: "global",
          path: "",
          rawPath: "",
          scope: "global",
          policyPath: "",
          icon: "🌐",
        },
      ];
    }
    renderSmartFolders(smartFolders);
    setSelectedFolder(smartFolders[0]);
  };

  knowledgeResultsList?.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.dataset.action;
    if (!action) {
      return;
    }
    const docPath = target.dataset.path || "";
    const docTitle = target.dataset.title || baseName(target.dataset.path || "");

    if (action === "copy") {
      if (docPath && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(docPath)
          .then(() => showStatus("문서 경로를 복사했습니다."))
          .catch(() => showStatus("경로 복사에 실패했습니다."));
      } else {
        showStatus("경로 복사 기능을 사용할 수 없습니다.");
      }
    } else if (action === "open") {
      if (window.toolbarAPI && typeof window.toolbarAPI.openPath === "function") {
        window.toolbarAPI
          .openPath(docPath)
          .then((result) => {
            if (result?.ok) {
              showStatus(`'${docTitle}' 문서를 여는 중입니다.`);
            } else {
              showStatus(result?.error || "문서를 열 수 없습니다.");
            }
          })
          .catch((err) => {
            console.error("[KnowledgeAgent] open error", err);
            showStatus("문서를 열 수 없습니다.");
          });
      } else {
        showStatus("문서 열기 기능을 사용할 수 없습니다.");
      }
    }
  });

  commandForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selectedFolder && smartFolders.length) {
      setSelectedFolder(smartFolders[0]);
    }
    const query = commandInput.value.trim();
    if (!query) {
      showStatus("요청할 내용을 입력해주세요.");
      return;
    }

    const intent = detectIntent(query);
    closePanels();
    console.log("[ToolbarCommand]", { query, intent, folder: selectedFolder });

    if (intent === "meeting") {
      runMeetingAgent(query);
    } else if (intent === "knowledge") {
      runKnowledgeAgent(query);
    } else {
      showStatus("어떤 작업인지 모르겠어요. ‘회의 요약’ 또는 ‘찾아줘’처럼 말씀해보세요.");
    }

    commandInput.value = "";
  });

  loadSmartFolders().then(() => updateCommandPlaceholder());
})();
