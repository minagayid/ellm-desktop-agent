from __future__ import annotations

from pathlib import Path


READABLE_SUFFIXES = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".xml", ".html", ".htm", ".log", ".pdf", ".docx"
}
OPENABLE_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".log", ".pdf", ".docx"}
WRITABLE_SUFFIXES = {".txt", ".md", ".csv", ".json"}
IGNORED_DIRS = {".git", ".venv", ".ssh", ".aws", ".azure", ".gnupg", "__pycache__", "node_modules", "models", "run-data"}
SENSITIVE_COMPONENTS = {".ssh", ".aws", ".azure", ".gnupg", ".netrc", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def is_sensitive_path(path: Path) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in SENSITIVE_COMPONENTS or lowered == ".env" or lowered.startswith(".env."):
            return True
        if lowered in {"credentials", "credentials.json", "secrets", "secrets.json", "tokens.json", "token.json", "service-account.json", "service_account.json"}:
            return True
    return path.suffix.casefold() in SENSITIVE_SUFFIXES


def scoped_path(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("path is outside the selected workspace")
    return resolved
