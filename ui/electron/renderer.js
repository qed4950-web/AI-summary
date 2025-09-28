(() => {
  const body = document.body;
  const toolbar = document.querySelector(".toolbar");
  const panelLayer = document.getElementById("panel-layer");
  const panel = document.getElementById("main-panel");
  const panelTitle = document.getElementById("panel-title");
  const panelSubtitle = document.getElementById("panel-subtitle");
  const closePanelBtn = document.getElementById("close-panel");
  const commandForm = document.getElementById("command-form");
  const commandInput = document.getElementById("command-input");
  const actionButtons = toolbar?.querySelectorAll(".icon[data-action]") ?? [];
  const viewButtons = toolbar ? Array.from(toolbar.querySelectorAll(".icon[data-view]")) : [];

  const viewPlaceholder = document.getElementById("view-placeholder");
  const viewSmartFolders = document.getElementById("view-smart-folders");
  const viewMeetingSummary = document.getElementById("view-meeting-summary");
  const viewKnowledgeResults = document.getElementById("view-knowledge-results");

  const smartFolderList = document.getElementById("smart-folder-list");
  const meetingSummaryFolder = document.getElementById("meeting-summary-folder");
  const meetingSummaryText = document.getElementById("meeting-summary-text");
  const meetingSummaryHighlights = document.getElementById("meeting-summary-highlights");
  const meetingSummaryActions = document.getElementById("meeting-summary-actions");
  const knowledgeResultsFolder = document.getElementById("knowledge-results-folder");
  const knowledgeResultsList = document.getElementById("knowledge-results-list");

  if (!toolbar || !panelLayer || !panel) {
    return;
  }

  let smartFolders = [];
  let selectedFolder = null;
  let micActive = false;
  let statusTimer = null;
  let activeView = "placeholder";

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const showStatus = (message) => {
    const status = toolbar.querySelector(".status");
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

  const togglePanel = (open, view = "placeholder", options = {}) => {
    if (open) {
      setActiveView(view, options);
      body.classList.add("is-expanded");
      panelLayer.classList.add("is-visible");
      panel.classList.add("is-visible");
      panelLayer.setAttribute("aria-hidden", "false");
    } else {
      panel.classList.remove("is-visible");
      panelLayer.setAttribute("aria-hidden", "true");
      setTimeout(() => {
        panelLayer.classList.remove("is-visible");
        body.classList.remove("is-expanded");
        setActiveView("placeholder");
      }, 280);
    }
  };

  const setActiveView = (view, { title, subtitle } = {}) => {
    activeView = view;
    viewPlaceholder.hidden = view !== "placeholder";
    viewSmartFolders.hidden = view !== "smart-folders";
    viewMeetingSummary.hidden = view !== "meeting-summary";
    viewKnowledgeResults.hidden = view !== "knowledge-results";

    const viewTitleMap = {
      placeholder: "Liquid Glass 패널",
      "smart-folders": "스마트 폴더",
      "meeting-summary": "회의 요약",
      "knowledge-results": "지식 검색 결과",
    };

    const viewSubtitleMap = {
      placeholder: "맥OS 스타일 패널",
      "smart-folders": "검색과 요약 범위를 선택하세요.",
      "meeting-summary": "AI가 생성한 회의 요약",
      "knowledge-results": "선택한 범위에서 찾은 결과",
    };

    panelTitle.textContent = title || viewTitleMap[view] || "패널";
    panelSubtitle.textContent = subtitle || viewSubtitleMap[view] || "";

    viewButtons.forEach((button) => {
      const matches = button.dataset.view === view;
      button.classList.toggle("is-active", matches);
    });
  };

  const updatePlaceholder = () => {
    if (!commandInput) {
      return;
    }
    if (!selectedFolder || (selectedFolder.scope || "auto") === "global") {
      commandInput.placeholder = "무엇을 도와드릴까요? 예: 회의록 요약해줘";
    } else {
      commandInput.placeholder = `${selectedFolder.label} 범위에서 명령을 입력하세요.`;
    }
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
    highlightSmartFolders();
    updatePlaceholder();
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

  const baseName = (targetPath = "") => {
    if (!targetPath) {
      return "문서";
    }
    const normalized = targetPath.replace(/\\/g, "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || targetPath;
  };

  const closePanelsForAction = () => {
    if (activeView !== "placeholder") {
      setActiveView("placeholder");
    }
  };

  const runMeetingAgent = async (query) => {
    closePanelsForAction();
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
        if (response?.ok) {
          updateMeetingSummary(response.data, response.fallback);
          togglePanel(true, "meeting-summary");
          showStatus("회의 요약 결과를 확인하세요.");
        } else {
          showStatus("회의 요약 중 오류가 발생했습니다.");
        }
      } catch (err) {
        showStatus("회의 요약 중 오류가 발생했습니다.");
      }
    }
  };

  const runKnowledgeAgent = async (query) => {
    closePanelsForAction();
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
        if (response?.ok) {
          updateKnowledgeResults(response.data, response.fallback);
          togglePanel(true, "knowledge-results");
          showStatus("검색 결과를 정리했습니다.");
        } else {
          showStatus("지식 검색 중 오류가 발생했습니다.");
        }
      } catch (err) {
        showStatus("지식 검색 중 오류가 발생했습니다.");
      }
    }
  };

  const updateMeetingSummary = (data, fallback) => {
    meetingSummaryFolder.textContent = fallback
      ? `${data?.folder?.label || selectedFolder?.label || "선택된 폴더"} (모의 데이터)`
      : data?.folder?.label || selectedFolder?.label || "선택된 폴더";
    meetingSummaryText.textContent = data?.summary || "요약 결과를 불러오지 못했습니다.";

    meetingSummaryHighlights.innerHTML = "";
    if (Array.isArray(data?.highlights) && data.highlights.length) {
      data.highlights.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        meetingSummaryHighlights.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "강조할 하이라이트가 없습니다.";
      meetingSummaryHighlights.appendChild(li);
    }

    meetingSummaryActions.innerHTML = "";
    if (Array.isArray(data?.actions) && data.actions.length) {
      data.actions.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        meetingSummaryActions.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "등록된 액션 아이템이 없습니다.";
      meetingSummaryActions.appendChild(li);
    }
  };

  const updateKnowledgeResults = (data, fallback) => {
    knowledgeResultsFolder.textContent = fallback
      ? `${data?.folder?.label || selectedFolder?.label || "선택된 폴더"} (모의 데이터)`
      : data?.folder?.label || selectedFolder?.label || "선택된 폴더";
    knowledgeResultsList.innerHTML = "";

    if (!Array.isArray(data?.items) || !data.items.length) {
      const li = document.createElement("li");
      li.textContent = "관련 문서를 찾지 못했습니다.";
      knowledgeResultsList.appendChild(li);
      return;
    }

    data.items.forEach((item) => {
      const li = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = item.title || baseName(item.path);
      const snippet = document.createElement("p");
      snippet.textContent = item.snippet || "미리보기 정보를 가져오지 못했습니다.";
      const path = document.createElement("p");
      path.textContent = item.path || "경로 정보 없음";
      path.classList.add("knowledge-results__folder");

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
  };

  const handleKnowledgeListClick = (event) => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target) {
      return;
    }
    const action = target.dataset.action;
    if (!action) {
      return;
    }
    const docPath = target.dataset.path || "";
    const docTitle = target.dataset.title || baseName(docPath);

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
          .catch(() => showStatus("문서를 열 수 없습니다."));
      } else {
        showStatus("문서 열기 기능을 사용할 수 없습니다.");
      }
    }
  };

  const detectIntent = (text) => {
    const normalized = text.toLowerCase();
    const meetingTriggers = ["회의", "회의록", "요약", "음성", "transcript"];
    const knowledgeTriggers = ["찾아", "검색", "어딨어", "보고", "summary"];
    if (meetingTriggers.some((word) => normalized.includes(word))) {
      return "meeting";
    }
    if (knowledgeTriggers.some((word) => normalized.includes(word))) {
      return "knowledge";
    }
    return null;
  };

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
      if (!data.length) {
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

  // Event bindings
  actionButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (action === "mic") {
        micActive = !micActive;
        button.classList.toggle("is-active", micActive);
        showStatus(micActive ? "음성 명령 대기 중" : "음성 캡처 일시중지");
      } else if (action === "settings") {
        showStatus("환경 설정 패널을 준비 중입니다.");
      }
    });
  });

  viewButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      if (!view) {
        return;
      }
      if (activeView === view && panelLayer.classList.contains("is-visible")) {
        togglePanel(false);
      } else if (view === "smart-folders") {
        togglePanel(true, "smart-folders");
      } else {
        togglePanel(true, view);
      }
    });
  });

  closePanelBtn?.addEventListener("click", () => togglePanel(false));

  panelLayer.addEventListener("click", (event) => {
    if (event.target === panelLayer) {
      togglePanel(false);
    }
  });

  commandForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!commandInput) {
      return;
    }
    const query = commandInput.value.trim();
    if (!query) {
      showStatus("요청할 내용을 입력해주세요.");
      return;
    }
    const intent = detectIntent(query);
    if (intent === "meeting") {
      runMeetingAgent(query);
    } else if (intent === "knowledge") {
      runKnowledgeAgent(query);
    } else {
      showStatus("어떤 작업인지 모르겠어요. ‘회의 요약’ 또는 ‘찾아줘’처럼 말씀해보세요.");
    }
    commandInput.value = "";
  });

  smartFolderList?.addEventListener("click", (event) => {
    const item = event.target instanceof HTMLElement ? event.target.closest("li") : null;
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
    showStatus(`${folder.label} 폴더를 선택했습니다.`);
  });

  knowledgeResultsList?.addEventListener("click", handleKnowledgeListClick);

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && panelLayer.classList.contains("is-visible")) {
      togglePanel(false);
    }
  });

  loadSmartFolders().then(() => updatePlaceholder());
})();
