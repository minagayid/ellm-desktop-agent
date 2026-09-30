from __future__ import annotations

from typing import Any


def bounded_setting(config: dict[str, Any], name: str, default: int, maximum: int, minimum: int = 1) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return max(minimum, min(value, maximum))
