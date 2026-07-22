# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   05/06/2026

"""Pytest fixtures shared across the test suite.

Provides a minimal ``yaml`` stand-in for environments where PyYAML isn't
importable, so config-loading tests keep working either way.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any


def _strip_yaml_comment(line: str) -> str:
    """Strip a simple YAML comment while respecting quoted strings."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    """Parse a tiny YAML scalar subset used by the tests."""
    text = value.strip()
    if text == "":
        return ""
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the simple top-level mappings/lists used in local tests."""
    data: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw_line = _strip_yaml_comment(lines[index]).rstrip()
        index += 1
        if not raw_line.strip():
            continue
        if raw_line[0].isspace() or ":" not in raw_line:
            raise YAMLError(f"Unsupported YAML line: {raw_line!r}")

        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value:
            data[key] = _parse_scalar(value)
            continue

        block_list: list[Any] = []
        block_dict: dict[Any, Any] = {}
        mode: str | None = None
        while index < len(lines):
            block_line = _strip_yaml_comment(lines[index]).rstrip()
            if not block_line.strip():
                index += 1
                continue
            if not block_line[0].isspace():
                break
            stripped = block_line.strip()
            index += 1
            if stripped.startswith("- "):
                if mode == "dict":
                    raise YAMLError(f"Mixed YAML block types under {key!r}")
                mode = "list"
                block_list.append(_parse_scalar(stripped[2:]))
            elif ":" in stripped:
                if mode == "list":
                    raise YAMLError(f"Mixed YAML block types under {key!r}")
                mode = "dict"
                item_key, item_value = stripped.split(":", 1)
                block_dict[_parse_scalar(item_key)] = _parse_scalar(item_value)
            else:
                raise YAMLError(f"Unsupported YAML block line: {stripped!r}")

        data[key] = block_list if mode == "list" else block_dict if mode == "dict" else None
    return data


class YAMLError(ValueError):
    """Minimal stand-in for PyYAML's YAMLError in tests."""


def _safe_dump(data: Any) -> str:
    """Serialize test data as JSON, which the fallback loader accepts."""
    return json.dumps(data)


def _safe_load(source: Any) -> Any:
    """Load JSON first, then the small YAML subset used by test fixtures."""
    text = source.read() if hasattr(source, "read") else source
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _simple_yaml_load(text)


if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_dump = _safe_dump
    yaml_module.safe_load = _safe_load
    yaml_module.YAMLError = YAMLError
    sys.modules["yaml"] = yaml_module
