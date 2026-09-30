from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .limits import bounded_setting
from .paths import IGNORED_DIRS, OPENABLE_SUFFIXES, READABLE_SUFFIXES, WRITABLE_SUFFIXES, is_sensitive_path, scoped_path


MUTATING_ACTIONS = {"write_file", "replace_file", "create_folder", "move_path", "run_command"}


class ToolExecutor:
    def __init__(self, workspace: Path, agent_config: dict[str, Any], commands: dict[str, dict[str, Any]]):
        self.workspace = workspace.resolve()
        self.agent_config = agent_config
        self.commands = commands

    def execute(self, action: str, arguments: dict[str, Any], *, approved: bool = False) -> tuple[str, bool]:
        if action in MUTATING_ACTIONS and not approved:
            raise PermissionError("this action requires an explicit approval")
        if action == "list_directory":
            return self.list_directory(arguments), False
        if action == "find_files":
            return self.find_files(arguments), False
        if action == "read_file":
            return self.read_file(arguments), False
        if action == "open_path":
            return self.open_path(arguments), False
        if action == "write_file":
            return self.write_file(arguments), True
        if action == "replace_file":
            return self.replace_file(arguments), True
        if action == "create_folder":
            return self.create_folder(arguments), True
        if action == "move_path":
            return self.move_path(arguments), True
        if action == "run_command":
            return self.run_command(arguments), True
        raise ValueError("action does not execute a tool")

    def _path(self, arguments: dict[str, Any]) -> Path:
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("path must be a non-empty string")
        return scoped_path(self.workspace, raw)

    def list_directory(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        if not path.is_dir():
            raise ValueError("folder does not exist")
        entries = []
        for item in sorted(path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.casefold()))[:100]:
            if is_sensitive_path(item.relative_to(self.workspace)):
                continue
            entries.append({"name": item.name, "kind": "folder" if item.is_dir() else "file"})
        return json.dumps({"path": str(path.relative_to(self.workspace)), "entries": entries}, ensure_ascii=False)

    def find_files(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        matches: list[str] = []
        for parent, dirs, files in os.walk(self.workspace):
            dirs[:] = [directory for directory in dirs if directory.casefold() not in IGNORED_DIRS]
            for name in files:
                if query.casefold() in name.casefold():
                    relative = (Path(parent) / name).relative_to(self.workspace)
                    if is_sensitive_path(relative):
                        continue
                    matches.append(str(relative))
                    if len(matches) == 100:
                        return json.dumps({"matches": matches, "truncated": True}, ensure_ascii=False)
        return json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False)

    def read_file(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        if is_sensitive_path(path.relative_to(self.workspace)):
            raise PermissionError("reading credential and key material is blocked")
        if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
            raise ValueError(f"unsupported or missing file; readable types: {', '.join(sorted(READABLE_SUFFIXES))}")
        size = path.stat().st_size
        max_file_bytes = bounded_setting(self.agent_config, "max_file_bytes", 2_000_000, 2_000_000)
        if size > max_file_bytes:
            raise ValueError("file is too large to pass to the local model")
        suffix = path.suffix.lower()
        cap = bounded_setting(self.agent_config, "max_observation_chars", 4_000, 4_000)
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError("install optional document readers with pip install -e '.[documents]'") from exc
            pieces: list[str] = []
            remaining = cap
            page_limit = bounded_setting(self.agent_config, "max_document_pages", 100, 100)
            for page_number, page in enumerate(PdfReader(str(path)).pages):
                if page_number >= page_limit or remaining <= 0:
                    break
                text = page.extract_text() or ""
                pieces.append(text[:remaining])
                remaining -= len(text)
            content = "\n".join(pieces)
        elif suffix == ".docx":
            try:
                from docx import Document
            except ImportError as exc:
                raise ValueError("install optional document readers with pip install -e '.[documents]'") from exc
            import zipfile

            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                expanded_bytes = sum(item.file_size for item in members)
                max_expanded = bounded_setting(self.agent_config, "max_docx_expanded_bytes", 20_000_000, 20_000_000)
                if expanded_bytes > max_expanded or len(members) > 2_000:
                    raise ValueError("Word document exceeds the safe extraction limits")
            pieces: list[str] = []
            remaining = cap
            for paragraph in Document(str(path)).paragraphs:
                if remaining <= 0:
                    break
                text = paragraph.text
                pieces.append(text[:remaining])
                remaining -= len(text) + 1
            content = "\n".join(pieces)
        else:
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                content = stream.read(cap + 1)
        return json.dumps(
            {"path": str(path.relative_to(self.workspace)), "content": content[:cap], "truncated": len(content) > cap},
            ensure_ascii=False,
        )

    def open_path(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        if is_sensitive_path(path.relative_to(self.workspace)):
            raise PermissionError("opening credential and key material is blocked")
        if not path.exists():
            raise ValueError("path does not exist")
        if path.is_file() and path.suffix.lower() not in OPENABLE_SUFFIXES:
            raise ValueError("open_path supports folders and supported non-web document formats only")
        if platform.system() == "Windows":
            os.startfile(str(path))  # noqa: S606 - opens only a validated path in its associated application.
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)], shell=False)
        else:
            subprocess.Popen(["xdg-open", str(path)], shell=False)
        return json.dumps({"opened": str(path.relative_to(self.workspace))}, ensure_ascii=False)

    def _file_content(self, arguments: dict[str, Any]) -> tuple[Path, str]:
        path = self._path(arguments)
        if path.suffix.lower() not in WRITABLE_SUFFIXES:
            raise ValueError(f"writable types: {', '.join(sorted(WRITABLE_SUFFIXES))}")
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        max_write_bytes = bounded_setting(self.agent_config, "max_write_bytes", 16_000, 16_000)
        if len(content.encode("utf-8")) > max_write_bytes:
            raise ValueError("new content is too large")
        return path, content

    def write_file(self, arguments: dict[str, Any]) -> str:
        path, content = self._file_content(arguments)
        if path.exists():
            raise FileExistsError("file already exists; use replace_file to request an approved replacement")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return json.dumps({"created": str(path.relative_to(self.workspace)), "bytes": len(content.encode("utf-8"))}, ensure_ascii=False)

    def replace_file(self, arguments: dict[str, Any]) -> str:
        path, content = self._file_content(arguments)
        if not path.is_file():
            raise ValueError("replacement target must be an existing regular file")
        backup = path.with_name(path.name + ".bak")
        if backup.exists():
            raise FileExistsError("backup already exists; refusing to overwrite it")
        path.rename(backup)
        try:
            path.write_text(content, encoding="utf-8", newline="\n")
        except Exception:
            backup.rename(path)
            raise
        return json.dumps(
            {"replaced": str(path.relative_to(self.workspace)), "backup": str(backup.relative_to(self.workspace))},
            ensure_ascii=False,
        )

    def create_folder(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        if path.exists():
            raise FileExistsError("folder path already exists")
        path.mkdir(parents=True, exist_ok=False)
        return json.dumps({"created_folder": str(path.relative_to(self.workspace))}, ensure_ascii=False)

    def move_path(self, arguments: dict[str, Any]) -> str:
        source_raw, destination_raw = arguments.get("source"), arguments.get("destination")
        if not isinstance(source_raw, str) or not source_raw.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(destination_raw, str) or not destination_raw.strip():
            raise ValueError("destination must be a non-empty string")
        source = scoped_path(self.workspace, source_raw)
        destination = scoped_path(self.workspace, destination_raw)
        if source == destination or not source.exists():
            raise ValueError("source must exist and differ from destination")
        if destination.exists():
            raise FileExistsError("destination already exists; refusing to overwrite it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return json.dumps(
            {"moved_from": str(source.relative_to(self.workspace)), "moved_to": str(destination.relative_to(self.workspace))},
            ensure_ascii=False,
        )

    def run_command(self, arguments: dict[str, Any]) -> str:
        command_id = arguments.get("command_id")
        if not isinstance(command_id, str) or command_id not in self.commands:
            raise ValueError("command name is not in the local allowlist")
        entry = self.commands[command_id]
        argv = entry.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv):
            raise ValueError("allowlisted command must define an argument array of strings")
        result = subprocess.run(
            argv,
            shell=False,
            cwd=self.workspace,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=bounded_setting(self.agent_config, "command_timeout_seconds", 20, 120),
            check=False,
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        output = output[: bounded_setting(self.agent_config, "max_command_output_chars", 4000, 4_000)]
        return json.dumps({"command_id": command_id, "return_code": result.returncode, "output": output}, ensure_ascii=False)
