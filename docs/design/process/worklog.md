# Meeting & Knowledge Agent Integration Log

## Todo Snapshot (2025-09-28)
- [ ] Add natural-language parser for toolbar input (meeting vs knowledge intent)
- [ ] Wire meeting agent pipeline call from Electron toolbar with smart folder context
- [ ] Display meeting summary/action items inside Electron UI
- [ ] Connect knowledge agent search to smart folders and show result list
- [ ] Expose smart folder picker in toolbar dashboard

---
## Progress 2025-09-28
- [x] Added toolbar command input + smart-folder launcher in Electron prototype (`ui/electron/index.html`, `style.css`, `renderer.js`).
- [x] Implemented placeholder intent routing for meeting / knowledge commands with status feedback.
- [x] Introduced smart folder selection sheet and state tracking.

Next up:
- Wire meeting agent placeholder to actual pipeline invocation via IPC/CLI.
- Display meeting summaries & search results in richer panel.
- Hook knowledge agent intent to meaningful search API.
---
### IPC Hook (2025-09-28)
- [x] Wired `run-meeting-agent` IPC channel in `ui/electron/main.js` with Python CLI fallback.
- [x] Added Electron preload/renderer glue to request meeting summaries and display panel.
- [x] Meeting summary panel renders folder info, summary text, and action items (placeholder for now).
- [x] Added mock meeting summary generator (scripts/pipeline/mock_meeting_summary.py) for Electron IPC testing.
- [x] Meeting summary panel now renders highlights & attendees placeholders from IPC response.
### Next Steps (Knowledge Agent)
- [ ] Implement `run-knowledge-agent` IPC channel with placeholder search results
- [ ] Render knowledge search results panel (document title, snippet, actions)
- [ ] Connect intent detection to actual search API (core/search/retriever or infopilot chat)
- [ ] Reconcile smart folder policy filtering for knowledge agent
- [x] Added `run-knowledge-agent` IPC hook returning mock search results.
- [x] Knowledge results panel in Electron renders folder, title, snippet, and path list.
### Knowledge Agent Integration Plan (2025-09-28)
- [ ] Replace mock search IPC with actual `infopilot.py chat --json` invocation (requires CLI JSON output)
- [ ] Parse search results into title/snippet/path objects for Electron panel
- [ ] Add result actions (open folder, copy path) with IPC bridge
- [ ] Handle failures/timeouts with user feedback
- [ ] Allow policy/scope wiring (smart folder -> policy agent)
### infopilot.py chat JSON TODO
- [x] Add `--json` flag to chat command for structured output
- [x] Define JSON schema (query, answer, suggestions, results[])
- [ ] Update Electron IPC to call `infopilot.py chat --json`
- [ ] Graceful fallback when CLI fails or JSON parsing errors
- [x] Electron knowledge IPC now calls `infopilot.py chat --query ... --json` with fallback mock on errors.
- [x] Knowledge results panel supports open (shell) and copy actions.
- [x] Smart folder selections now carry scope/policy metadata for IPC calls.
- [x] open-path handler resolves relative paths against repo root before invoking shell.
- [x] Electron loads smart folders from `core/config/smart_folders.json`; default folders seeded under `~/Documents/AI Summary/...`.
- [x] Added `scripts/util/init_workspace.py` to create default directories based on config.
