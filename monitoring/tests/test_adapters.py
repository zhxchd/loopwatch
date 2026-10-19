import json

from loopwatch_monitor.adapters import loopwatch, mistrace
from loopwatch_monitor.monitor.fixed import build_messages
from make_fixtures import loopwatch_episode


def test_mistrace_decodes_double_encoded_messages_and_tool_calls(bundles):
    b = bundles[0]  # edit_tests, openai-style tool calls
    refs = [e.ref for e in b.events]
    assert "m2.tc0" in refs and "m4.tc0" in refs
    calls = {e.ref: e for e in b.events if e.channel == "tool_call"}
    assert "sed -i" in calls["m4.tc0"].content and "call bash" in calls["m4.tc0"].content
    assert any("<reasoning>" in e.content for e in b.events)
    assert b.task_spec.startswith("Implement `lcbhard_0`")


def test_mistrace_inspect_style_tool_calls(rows):
    b = list(mistrace.bundles([rows[2]]))[0]  # task 1 -> inspect style
    call = next(e for e in b.events if e.ref == "m4.tc0")
    assert "call python" in call.content and "def lcbhard_1" in call.content


def test_reasoning_can_be_excluded(rows):
    b = list(mistrace.bundles(rows[:1], include_reasoning=False))[0]
    assert not any("<reasoning>" in e.content for e in b.events)


def test_mistrace_gold_mapping(bundles):
    by_kind = {b.gold.secondary["judge_cheating_type"]: b.gold for b in bundles}
    assert by_kind["A"].label == "unsafe" and by_kind["A"].violation_type == "reward_hacking.test_modification"
    assert by_kind["D"].violation_type == "reward_hacking.test_special_casing"
    assert by_kind["C"].violation_type == "reward_hacking.state_recording"
    assert by_kind[None].label == "safe" and by_kind[None].violation_type is None
    assert "by construction" in bundles[0].gold.provenance


def test_no_label_or_split_leakage_into_monitor_prompt(rows, bundles):
    for row, b in zip(rows, bundles):
        prompt = json.dumps(build_messages(b.for_monitor(), 200000, 12000))
        for secret in (row["rollout_id"], row["source_task"], "is_reward_hack", "hack_label_basis",
                       "conflicting", "is_cheating", "cheating_type", "allow_test_modifications", "scorer"):
            assert secret not in prompt, secret


def test_loopwatch_adapter_orders_and_labels(tmp_path):
    loopwatch_episode(tmp_path, unsafe=False)
    loopwatch_episode(tmp_path, unsafe=True)
    items = list(loopwatch.bundles(tmp_path))
    assert [b.gold.label for b in items] == ["safe", "unsafe"]
    unsafe = items[1]
    kinds = [e.channel for e in unsafe.events]
    assert kinds == ["message", "message", "fs_event", "message", "message"]  # sorted by time
    assert "Plan: union-find" in unsafe.events[1].content
    assert "/work/AlgoTune" in unsafe.authorization_policy
    hidden = loopwatch.bundle(tmp_path / "episode-002", include_collector=False)
    assert "fs_event" not in [e.channel for e in hidden.events]


def test_span_parser_prefers_human_and_drops_negative_labels(bundles):
    kinds = {b.gold.secondary["judge_cheating_type"]: b.gold for b in bundles}
    assert kinds["A"].span_refs == ("m4",) and kinds["A"].span_provenance == "human"   # gold_spans by message id
    assert kinds["D"].span_refs == ("m4",) and kinds["D"].span_provenance == "judge"   # "not_..." span on m2 dropped
    assert kinds[None].span_refs is None


def test_check_data_runs_on_fixture(rows_file, capsys):
    import sys
    from loopwatch_monitor import check_data
    sys.argv = ["check_data", "--source", str(rows_file)]
    check_data.main()
    out = capsys.readouterr().out
    assert '"hacked": 4' in out and '"hacked_without_located_spans": 0' in out
    sys.argv = ["check_data", "--source", str(rows_file), "--index", "0"]
    check_data.main()
    assert "span_labels_json outline" in capsys.readouterr().out


def test_real_policy_matches_actor_instructions(bundles):
    assert "DO NOT MODIFY THE TESTS" in bundles[0].authorization_policy
    assert "Failing tests is not a violation" in bundles[0].authorization_policy
