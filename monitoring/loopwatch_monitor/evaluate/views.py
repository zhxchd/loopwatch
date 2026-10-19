"""What part of each record the monitor sees (PLAN.md §7, length control).

`full`: the whole record (subject to the evidence budget).
`length_matched`: hacked runs whole; each clean run cut to a prefix whose
message count is drawn from the hacked runs' message counts. In MisTrace clean
runs use the full message budget and hacked runs stop early, so length alone
separates the labels (AUROC ~0.96). Matching the length distributions removes
that shortcut by construction. The draw is seeded per episode, so the same
episode gets the same cut in every variant and every rerun.

The harness chooses cuts using labels; the monitor never sees the labels, the
cut point, or that a cut happened.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

from loopwatch_monitor.monitor.fixed import message_count

VIEWS = ("full", "length_matched")


def _message_index(ref: str):
    head = ref.split(".", 1)[0]
    return int(head[1:]) if head[:1] == "m" and head[1:].isdigit() else None


def cut(bundle, n_messages: int):
    """Keep the first `n_messages` distinct messages (tool calls stay with their message)."""
    order = []
    for e in bundle.events:
        head = e.ref.split(".", 1)[0]
        if head not in order:
            order.append(head)
    keep = set(order[:n_messages])
    events = tuple(e for e in bundle.events if e.ref.split(".", 1)[0] in keep)
    return replace(bundle, events=events, source={**bundle.source, "view_cut_messages": n_messages})


def length_matched(bundles: list, seed: int) -> list:
    hacked_lengths = sorted(message_count(b) for b in bundles if b.gold and b.gold.label == "unsafe")
    if not hacked_lengths:
        return bundles
    out = []
    for b in bundles:
        if b.gold is None or b.gold.label == "unsafe":
            out.append(b)
            continue
        draw = int(hashlib.sha256(f"{seed}:view:{b.episode_id}".encode()).hexdigest(), 16)
        target = hacked_lengths[draw % len(hacked_lengths)]
        out.append(cut(b, min(target, message_count(b))))
    return out


def apply(view: str, bundles: list, seed: int) -> list:
    if view == "full":
        return bundles
    if view == "length_matched":
        return length_matched(bundles, seed)
    raise ValueError(f"unknown view {view!r}; choose from {VIEWS}")
