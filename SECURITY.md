# Security notes

This project gives a language model a small set of local tools. The model is not a security boundary.

- File actions are restricted to the workspace root selected at startup. Resolved paths, including symlinks, must remain beneath that root.
- The agent can list, search, read, open, create, replace, create folders, and move supported paths. It has no delete action. Credential/key paths are blocked from read/open/move operations, including attempts to rename them to ordinary-looking filenames. Replacing an existing file requires explicit confirmation and creates a `.bak` copy first.
- A command can run only by a name present in the local command registry. The agent cannot supply executable text or arguments. The user must confirm each command. Commands run with `shell=False`, a fixed workspace, and a timeout. A configured executable may still be powerful; review every local command entry before use.
- Malformed output, unknown tools, invalid arguments, paths outside the workspace, and missing command names stop without executing anything.
- If the user declines a mutating action and the model repeats that plan, the host stops instead of asking again; path arguments are normalized before comparing plans.
- Documents and command output are untrusted data. They never change the tool allowlist or approval policy. Prompt wording is not a complete defense against instruction injection, so keep the workspace limited to material the agent may inspect.
- Text extraction is bounded by file size, output length, PDF page count, and DOCX expanded archive size. Generated writes are capped at 16 KB and the full proposed text is shown before approval.
- The model downloads from Hugging Face on first use. Prompts and local files are passed to that downloaded local model; inference is not sent to a hosted API. Network access is needed for initial package/model downloads only.
- Run only in folders you intend the agent to inspect. Do not add secrets or private folders to the allowed workspace.
- Training is local. Candidate weights and runtime logs are ignored by Git by default. Do not publish weights or data unless you have reviewed them.

Report security issues privately to the repository owner before disclosing them publicly.
