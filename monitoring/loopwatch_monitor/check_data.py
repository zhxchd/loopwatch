"""Check real MisTrace rows before spending API money. No model calls.

    python -m loopwatch_monitor.check_data --source data/mistrace-reward-hacking.parquet
    python -m loopwatch_monitor.check_data --source data/mistrace-reward-hacking.parquet --index 0

Answers three questions:
  1. Did the adapter decode messages and tool calls?
  2. Could the span parser locate the labelled hacking steps (for grounding)?
  3. Do the actor's own instructions mention editing tests? If they permit it,
     the monitor's policy ("never edit tests") disagrees with what the actor
     was told, which matters when reading false alarms and misses.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

from loopwatch_monitor.adapters import mistrace
from loopwatch_monitor.adapters.common import decode, text_of
from loopwatch_monitor.evaluate.metrics import auroc
from loopwatch_monitor.evaluate import views
from loopwatch_monitor.monitor.fixed import last_tests_passed, message_count

TEST_PERMISSION = re.compile(r"(modif|edit|chang|updat|delet|rewrit)\w*\s+(\w+\s+){0,3}tests?\b|tests?\s+(\w+\s+){0,3}(may|can|cannot|must not|should not)", re.I)


def shape(value, depth=0, max_depth=4):
    """Key/type outline of a JSON value (first element of lists only)."""
    pad = "  " * depth
    if depth > max_depth:
        return pad + "..."
    if isinstance(value, dict):
        lines = []
        for k, v in list(value.items())[:25]:
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}: {type(v).__name__}" + (f"[{len(v)}]" if isinstance(v, list) else ""))
                lines.append(shape(v, depth + 1, max_depth))
            else:
                lines.append(f"{pad}{k}: {type(v).__name__} = {json.dumps(v)[:80]}")
        return "\n".join(lines)
    if isinstance(value, list):
        return shape(value[0], depth, max_depth) if value else pad + "(empty)"
    return pad + json.dumps(value)[:120]


def shortcut_check(bundles) -> dict:
    """AUROC of the two no-skill shortcuts on a given view (0.5 = no signal)."""
    label = lambda b: {"label": b.gold.label}
    return {"length_auroc": round(auroc([{"gold": label(b), "assessment": {"valid": True, "risk_score": -message_count(b)}}
                                         for b in bundles]), 3),
            "last_test_run_passed_auroc": round(auroc([{"gold": label(b), "assessment": {"valid": True, "risk_score": last_tests_passed(b)}}
                                                       for b in bundles]), 3)}


def instructions(row) -> str:
    messages = decode(row.get("messages_json")) or []
    return "\n".join(text_of(m.get("content"), False) for m in messages
                     if isinstance(m, dict) and m.get("role") in ("system", "user"))[:20000]


def one(row, index):
    b = next(mistrace.bundles([row]))
    channels = Counter(e.channel for e in b.events)
    print(f"=== row {index}: {b.episode_id}")
    print(f"label={b.gold.label} type={b.gold.violation_type} judge_type={row.get('cheating_type')} model={row.get('model')}")
    print(f"events={len(b.events)} {dict(channels)}")
    print(f"span refs ({b.gold.span_provenance}): {b.gold.span_refs}")
    print("\n--- span_labels_json outline ---")
    print(shape(decode(row.get("span_labels_json"))))
    text = instructions(row)
    print("\n--- actor instructions mentioning tests ---")
    hits = [m.group(0) for m in TEST_PERMISSION.finditer(text)]
    print("\n".join(f"  ...{text[max(0, text.find(h) - 120):text.find(h) + 160]!r}" for h in hits[:4]) or "  (none)")
    print("\n--- task spec given to the monitor ---")
    print(b.task_spec[:1500])


def aggregate(rows):
    bundles = list(mistrace.bundles(rows))
    hacked = [b for b in bundles if b.gold.label == "unsafe"]
    sizes = sorted(sum(len(e.content) for e in b.events) for b in bundles)
    mention = [bool(TEST_PERMISSION.search(instructions(r))) for r in rows]
    by_label = Counter((b.gold.label, m) for b, m in zip(bundles, mention))
    counts = [message_count(b) for b in bundles]
    lengths = {label: sorted(c for c, b in zip(counts, bundles) if b.gold.label == label) for label in ("unsafe", "safe")}
    length_rows = [{"gold": {"label": b.gold.label}, "assessment": {"valid": True, "risk_score": -c}}
                   for c, b in zip(counts, bundles)]
    print(json.dumps({
        "rows": len(rows),
        "length_shortcut_auroc": round(auroc(length_rows), 3) if hacked and len(hacked) < len(bundles) else None,
        "last_test_run_passed_shortcut_auroc": round(auroc([{"gold": {"label": b.gold.label}, "assessment": {"valid": True, "risk_score": last_tests_passed(b)}} for b in bundles]), 3) if hacked and len(hacked) < len(bundles) else None,
        "length_matched_view": shortcut_check(views.length_matched(bundles, 42)),
        "runs_that_wrote_test_files": dict(Counter(f"{b.gold.label}" for b in bundles if b.gold.secondary.get("writes_test_file"))),
        "median_messages": {k: v[len(v) // 2] if v else None for k, v in lengths.items()},
        "long_runs_ge_90_messages": {k: sum(c >= 90 for c in v) for k, v in lengths.items()}, "hacked": len(hacked), "clean": len(bundles) - len(hacked),
        "columns": sorted(rows[0].keys()) if rows else [],
        "hacked_with_located_spans": Counter(b.gold.span_provenance for b in hacked if b.gold.span_refs),
        "hacked_without_located_spans": sum(not b.gold.span_refs for b in hacked),
        "rows_with_no_events": sum(not b.events for b in bundles),
        "record_chars": {"median": sizes[len(sizes) // 2] if sizes else None, "max": sizes[-1] if sizes else None},
        "instructions_mention_editing_tests": {f"{label}/{'mentions' if m else 'silent'}": n for (label, m), n in sorted(by_label.items())},
        "tool_settings": Counter(str(r.get("tool_setting")) for r in rows),
        "models": Counter(str(r.get("model")) for r in rows),
    }, indent=2, default=dict))
    if hacked and not any(b.gold.span_refs for b in hacked):
        print("\nNo spans located: run with --index on a hacked row and share the outline so the parser can be fixed. "
              "Grounding rows in the report will read n/a until then; detection metrics are unaffected.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    rows = mistrace.load_rows(args.source)
    if args.index is not None:
        one(rows[args.index], args.index)
    else:
        aggregate(rows)


if __name__ == "__main__":
    main()
