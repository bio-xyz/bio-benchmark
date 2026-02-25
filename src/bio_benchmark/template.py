from __future__ import annotations

import re
from typing import Any

TEMPLATE_TOKEN = re.compile(r"{{\s*([a-zA-Z0-9_.-]+)\s*}}")


def render_template_string(template: str, context: dict[str, Any]) -> str:
    """Render a tiny {{token}} template string with values from context."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key, "")
        return str(value)

    return TEMPLATE_TOKEN.sub(_replace, template)


def render_template_data(data: Any, context: dict[str, Any]) -> Any:
    if isinstance(data, str):
        return render_template_string(data, context)
    if isinstance(data, dict):
        return {key: render_template_data(value, context) for key, value in data.items()}
    if isinstance(data, list):
        return [render_template_data(item, context) for item in data]
    return data


def extract_nested_value(payload: Any, dot_path: str) -> Any:
    value: Any = payload
    for raw_part in dot_path.split("."):
        part = raw_part.strip()
        if not part:
            continue
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        return None
    return value

