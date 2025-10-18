# UI Changelog

## 2024-11-24
- Mac-style liquid glass refresh for Electron toolbar and panels (glass gradients, blur, glow accents).
- Re-centered panel overlay with masked backdrop light ring to avoid dark tint behind the toolbar.
- Responsive panel sizing tuned (vw/vh caps) with softer fade/slide animations.
- Toolbar remains pill-sized at bottom; icon hovers use mint/blue glow.
- Added AI assistant chat panel wired to LNPChat backend (IPC `chat-llm`), including suggestion chips and toolbar entry.
- Introduced training center panel for global vs smart-folder learning modes; training requests queue via `train_agent.py`.
- Training center now polls `training_status.json` and shows job logs/status, with IPC `training-status` and history view.
