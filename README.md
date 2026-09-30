# ELLM Desktop Agent

A local, text-first assistant for everyday device work: open folders and documents, find and read files, summarize material, draft or update text files, and run named commands you have explicitly added to its local allowlist.

The runtime uses your Hugging Face adapter, `minagayid/ELLM`, on its SmolLM2-135M-Instruct base as the planner. It does not call a hosted inference API. The first run downloads the public adapter and base checkpoint to the local Hugging Face cache. The project also includes a local LoRA fine-tuning path for bilingual English/Arabic task routing.

## What this version does

- Uses ELLM to choose one structured action at a time, observe its result, then plan the next step.
- Supports listing/searching a selected workspace, reading common text/config formats plus PDF/DOCX, opening files or folders, creating folders and text files, moving paths, replacing a text file with a backup, and summarizing content.
- Runs only named command entries from `config/commands.local.json`. Each command requires a confirmation. No free-form shell command is exposed to the model.
- Requires confirmation for creating/replacing files, creating folders, moving paths, and running commands. It never deletes files.
- Rejects malformed plans, unknown tools, invalid paths, common credential/key paths, or workspace escapes without executing an action. Mutations also require an executor approval flag and visible CLI confirmation.
- Stores no conversation history by default. A selected document and the current task context are sent only to the locally downloaded model.

## Model limits

The current `minagayid/ELLM` adapter is a small experimental LoRA adapter on SmolLM2-135M-Instruct, not a general desktop agent model. Its current model card reports 30.9% exact accuracy on a synthetic evidence-triage task and a 92.2% false-`RELEASE` rate for non-release examples; that benchmark is unrelated to desktop tasks, but it is a clear warning against trusting the model to enforce policy. Hugging Face also describes the SmolLM2 family as primarily English, so Arabic performance from this small fine-tune is unproven. This project never uses the model's output as permission. It validates each action independently and requires human confirmation for changes and commands.

This build is an experimental prototype, not a dependable daily-use agent. In the project's held-out benchmark, the existing adapter returned valid JSON for 0/88 cases, and the local candidate returned valid JSON for 30/88 while making unsafe choices on all 8 critical cases. The candidate failed the published gate and was not promoted. See [TRAINING_RESULTS.md](TRAINING_RESULTS.md). The included synthetic curriculum is small and does not establish broad reliability. No remote or paid training is used.

## Requirements

- Windows 10/11, Python 3.11, and an NVIDIA GPU for local training. Inference also works on CPU, but will be slower.
- An NVIDIA GPU with around 8 GB VRAM and 16 GB system RAM is the target for the bounded training run. Actual speed and memory use depend on context length and other applications.
- Internet access for the initial dependency/model downloads. After setup, inference and the built-in file tools are local.

## Install

Open PowerShell in this folder. The setup script makes a project-local virtual environment and installs PyTorch from its official CUDA 12.8 wheel index, then installs this project. It does not install Ollama, require administrator access, or download model weights until first use.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
./install.ps1
```

To enable PDF and Word document extraction, install the optional readers:

```powershell
./.venv/Scripts/python.exe -m pip install -e '.[documents]'
```

## Run

Choose a workspace folder the agent may inspect. All file paths are resolved beneath that folder, including symlinks.

```powershell
./run.ps1 -Workspace 'C:\Users\<you>\Documents\AgentWorkspace'
```

Or invoke the installed command directly:

```powershell
./.venv/Scripts/ellm-agent.exe --workspace 'C:\Users\<you>\Documents\AgentWorkspace'
```

Example requests:

- “Find the report about the Q3 budget and summarize it in Arabic.”
- “Open the meeting notes folder and list the files.”
- “Write a short summary to `summary.md`.”
- “Run the command that lists my local Ollama models.”

The agent displays the full proposed text (up to 16 KB) or exact move paths, then waits for `y` before changing anything or running a command. Read/open operations within the selected workspace do not require confirmation, but common credential and private-key paths are blocked. `Ctrl+C` stops the run.

## Configure named commands

Copy `config/commands.example.json` to `config/commands.local.json`, then add entries with a fixed executable and argument list. The local file is ignored by Git. The model can select only a command name and cannot generate or alter the executable or arguments. Review commands before adding them; each execution still requires approval.

```json
{
  "show_python_version": {
    "argv": ["python", "--version"],
    "description": "Show the installed Python version",
    "approval_required": true
  }
}
```

Commands run with no shell, a fixed working folder, captured output, a timeout, and bounded output. Avoid configuring commands that delete files, install software, change account settings, transmit data, or require administrator access.

## Fine-tune ELLM for task following

The training pipeline continues the existing LoRA adapter; it does not train a foundation model from scratch or modify your Hugging Face repository. The built-in dataset contains English and Arabic examples for file actions, summaries, safe command aliases, ambiguous requests, and refusal/clarification cases. You can inspect and edit `training/data/build_dataset.py` before generating the splits.

```powershell
./.venv/Scripts/python.exe training/data/build_dataset.py
./.venv/Scripts/python.exe training/train_agent_adapter.py --epochs 2
./.venv/Scripts/python.exe training/evaluate_agent_adapter.py --adapter-dir models/candidates/latest
```

Training requires a compatible CUDA PyTorch install and downloads the adapter/base from Hugging Face if they are not cached. Each run creates a separate candidate under `models/candidates/`; it refuses to overwrite one. The evaluation report measures structured-action accuracy, valid JSON rate, exact tool-plan accuracy (including paths, aliases, and requested file text), and unsafe choices on fixed held-out examples. The held-out set has 88 templated synthetic examples, so one answer changes the aggregate by about 1.1 percentage points. A synthetic score is not evidence that the model is safe or reliable in general. Review the report before selecting any candidate with `-Adapter`; the candidate from this run failed its gate and should not be selected.

The model adapter, base model, and source code have separate licensing. ELLM and its SmolLM2 base are Apache-2.0; see the linked model cards. This repository's source license is MIT. Review model/data terms before redistributing trained weights.

## Project map

- `ellm_agent/` — local inference, prompt, strict action validation, workspace-bound tools, and interactive CLI.
- `config/` — public examples; local settings are kept out of Git.
- `training/data/` — bilingual synthetic instruction curriculum and split builder.
- `training/` — local LoRA continuation and held-out action evaluation.
- `TRAINING_RESULTS.md` — measured losses and held-out results for this local run.
- `install.ps1`, `run.ps1` — Windows setup and launch scripts.
- `SECURITY.md` — tool boundaries and data handling.

The repository source is licensed under MIT. The `minagayid/ELLM` adapter and SmolLM2 base are Apache-2.0; review their model cards and terms separately before redistributing any trained weights.

This prototype is text-first. Wake-word detection, ASR/TTS, broad desktop clicking, web browsing, automatic memory extraction, and autonomous model promotion are out of scope for this first version.

## References

- [ELLM model card](https://huggingface.co/minagayid/ELLM)
- [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [Hugging Face PEFT adapter loading and training](https://huggingface.co/docs/transformers/peft)
