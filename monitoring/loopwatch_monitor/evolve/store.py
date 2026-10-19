"""Versioned prompt/skill store for the evolving monitor (PLAN.md §8). Seam only.

The baseline is version "v0" with an empty store: the fixed monitor reads no
skills and writes nothing. The evolving monitor will append versions here, each
carrying the dev-split score that admitted it; versions are never edited.
"""
import json
from pathlib import Path


class Store:
    def __init__(self, path):
        self.path = Path(path)

    def versions(self) -> list:
        if not self.path.exists():
            return [{"version": "v0", "skills": [], "prompt_patch": None, "admitted_by": "baseline"}]
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def current(self) -> dict:
        return self.versions()[-1]
