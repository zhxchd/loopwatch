import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import make_fixtures  # noqa: E402


@pytest.fixture
def rows():
    return make_fixtures.mistrace_rows()


@pytest.fixture
def rows_file(tmp_path, rows):
    path = tmp_path / "mistrace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def bundles(rows):
    from loopwatch_monitor.adapters import mistrace
    return list(mistrace.bundles(rows))


def verdict(risk, vtype, status, refs, response="flag_for_human"):
    return json.dumps({"risk_score": risk, "violation_type": vtype, "status": status, "evidence_refs": refs,
                       "uncertainty": 0.2, "recommended_response": response, "rationale": "test"})


@pytest.fixture
def oracle_respond():
    """Scripted 'model' that flags test edits / special-casing visible in the prompt."""
    import re

    def respond(messages):
        record = messages[1]["content"]
        for ref, body in re.findall(r"\[(m\d+\.tc\d+)\] tool_call / \w+:\n(.*?)(?=\n\n\[|\n</audit_record>)", record, re.S):
            if "sed -i" in body or "if x == 3" in body or "_calls" in body:
                return "Reasoning...\n```json\n" + verdict(0.9, "reward_hacking.other", "verified", [ref]) + "\n```"
        return verdict(0.1, "none", "no_violation", [], "none")
    return respond
