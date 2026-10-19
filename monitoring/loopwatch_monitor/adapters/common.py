"""Helpers shared by adapters."""
import json
from typing import Any


def decode(value: Any, depth: int = 3) -> Any:
    """MisTrace stores JSON as (sometimes doubly) encoded strings; unwrap them."""
    for _ in range(depth):
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{\"":
            return value
        try:
            value = json.loads(stripped)
        except ValueError:
            return value
    return value


def text_of(content: Any, include_reasoning: bool = True) -> str:
    """Flatten string / content-block message bodies into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "reasoning" or "reasoning" in block and "text" not in block:
            if include_reasoning and not block.get("redacted"):
                parts.append("<reasoning>\n" + str(block.get("reasoning") or block.get("summary") or "") + "\n</reasoning>")
        elif "text" in block:
            parts.append(str(block["text"]))
        elif kind in ("image", "audio", "video"):
            parts.append(f"[{kind} omitted]")
        else:
            parts.append(json.dumps(block, default=str)[:2000])
    return "\n".join(p for p in parts if p)


def arguments_text(arguments: Any) -> str:
    arguments = decode(arguments)
    if isinstance(arguments, dict):
        # Show code/commands literally rather than JSON-escaped.
        lines = []
        for key, value in arguments.items():
            value = value if isinstance(value, str) else json.dumps(value, default=str)
            lines.append(f"{key}:\n{value}" if "\n" in value else f"{key}: {value}")
        return "\n".join(lines)
    return arguments if isinstance(arguments, str) else json.dumps(arguments, default=str)
