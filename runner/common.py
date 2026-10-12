"""Small shared artifact helpers; never serializes environment variables."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read_events(path):
    events = []
    malformed = 0
    if not Path(path).exists():
        return events, 1
    for line in Path(path).read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            malformed += 1
    return events, malformed
