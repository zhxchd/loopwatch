import pytest

from conftest import verdict
from loopwatch_monitor.schema import extract_json_object, parse_assessment


def test_valid_assessment_parses(bundles):
    b = bundles[0].for_monitor()
    ref = sorted(b.refs())[0]
    a = parse_assessment("thinking...\n" + verdict(0.8, "reward_hacking.test_modification", "verified", [ref]), b)
    assert a.valid and a.risk_score == 0.8 and a.evidence_refs == [ref]


@pytest.mark.parametrize("text, error", [
    (verdict(0.8, "reward_hacking.other", "verified", ["m999"]), "unknown events"),
    (verdict(0.8, "reward_hacking.other", "verified", []), "must cite"),
    (verdict(1.4, "none", "no_violation", []), "risk_score"),
    (verdict(0.5, "made_up_type", "suspected", ["m0"]), "made_up_type"),
    (verdict(0.5, "none", "suspected", ["m0"]), "needs a violation_type"),
    ('{"risk_score": 0.3}', "missing fields"),
    ("no json at all", "no JSON"),
    ("", "empty"),
])
def test_bad_outputs_become_explicit_invalid(bundles, text, error):
    a = parse_assessment(text, bundles[0].for_monitor())
    assert not a.valid and a.risk_score is None and error in a.error
    assert a.status == "insufficient_evidence"


def test_extract_last_json_object_with_braces_in_prose():
    text = 'Code like {x: 1} appears.\n{"a": {"b": 2}}'
    assert extract_json_object(text) == {"a": {"b": 2}}


def test_for_monitor_strips_gold_ids_and_future_events(bundles):
    b = bundles[0]
    assert b.gold is not None
    m = b.for_monitor(checkpoint=2)
    assert m.gold is None
    assert set(m.source) <= {"dataset", "subset"}
    assert all(e.idx <= 2 for e in m.events) and len(m.events) == 3


def test_message_level_citation_of_tool_call_only_message_is_valid():
    from loopwatch_monitor.schema import AuditBundle, EvidenceItem
    b = AuditBundle("x", {}, "", "", (EvidenceItem("m82.tc0", 0, "tool_call", "assistant", "edit"),
                                      EvidenceItem("m83", 1, "tool_result", "tool", "ok")))
    for refs in (["m82"], ["m83.tool_result"], ["m82.tc0"]):
        assert parse_assessment(verdict(0.9, "reward_hacking.other", "verified", refs), b).valid, refs
    assert not parse_assessment(verdict(0.9, "reward_hacking.other", "verified", ["m84"]), b).valid
