# Changelog

## 0.1.0 — initial local prototype
- Adds a Windows-first, offline-first task agent backed by the user's ELLM Hugging Face adapter.
- Limits actions to workspace-scoped file tools and named, user-configured commands.
- Requires approval for file writes, folder creation, moves, and commands; blocks common credential paths from reading.
- Adds an English/Arabic instruction-tuning workflow with train/dev/test separation and a fail-closed evaluation gate.
