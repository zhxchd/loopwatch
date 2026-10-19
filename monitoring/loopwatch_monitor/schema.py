"""Normalized audit record and monitor assessment (PLAN.md §4-5).

Every adapter emits an AuditBundle; every monitor consumes one and returns a
MonitorAssessment. Gold labels live on the bundle for the scorer only and are
removed by `for_monitor()` before any monitor call.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Optional

from loopwatch_monitor.taxonomy import ViolationType, family


class Channel(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FS_EVENT = "fs_event"
    DOCKER_API = "docker_api"
    VERIFIER = "verifier"


class Status(str, Enum):
    VERIFIED = "verified"                    # concrete evidence an unsafe action happened
    SUSPECTED = "suspected"                  # intent or partial evidence, not confirmed
    NO_VIOLATION = "no_violation"            # evidence reviewed, nothing unsafe
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Response(str, Enum):
    # Passive mode acts only on NONE / FLAG_FOR_HUMAN; the rest are recorded but
    # inert so the same monitor can be reused unchanged in the active condition.
    NONE = "none"
    FLAG_FOR_HUMAN = "flag_for_human"
    REQUEST_EVIDENCE = "request_evidence"
    WARN = "warn"
    RECOMMEND_PAUSE = "recommend_pause"
    RECOMMEND_TERMINATE = "recommend_terminate"


@dataclass(frozen=True)
class EvidenceItem:
    ref: str          # stable id the monitor must cite, e.g. "m12" or "m12.tc0"
    idx: int          # position in the ordered audit record
    channel: str
    actor: str        # assistant | tool | system | user | collector
    content: str
    t: Optional[float] = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Gold:
    label: str                         # "unsafe" | "safe"
    violation_type: Optional[str]      # ViolationType value, or None when safe
    provenance: str                    # how the label was produced (must be non-monitor)
    secondary: dict = field(default_factory=dict)   # e.g. judge labels, kept apart
    # Message-level refs ("m4") where annotators marked the violation; None = unavailable.
    span_refs: Optional[tuple] = None
    span_provenance: Optional[str] = None           # "human" | "judge"


@dataclass(frozen=True)
class AuditBundle:
    episode_id: str
    source: dict
    task_spec: str
    authorization_policy: str
    events: tuple
    group: str = ""                    # split key: near-duplicates share a group
    checkpoints: tuple = ()            # event indices; empty means final only
    gold: Optional[Gold] = None

    def for_monitor(self, checkpoint: Optional[int] = None) -> "AuditBundle":
        """Strip gold and anything after the checkpoint (no future events)."""
        events = self.events if checkpoint is None else tuple(e for e in self.events if e.idx <= checkpoint)
        return replace(self, gold=None, events=events,
                       source={k: v for k, v in self.source.items() if k in ("dataset", "subset")})

    def refs(self) -> set:
        return {e.ref for e in self.events}


@dataclass
class MonitorAssessment:
    episode_id: str
    risk_score: Optional[float]
    violation_type: str
    status: str
    evidence_refs: list
    uncertainty: Optional[float]
    recommended_response: str
    rationale: str = ""
    valid: bool = True                 # False when the output could not be parsed
    error: Optional[str] = None
    usage: dict = field(default_factory=dict)   # tokens, cost_usd, latency_s
    monitor_version: str = "v0"
    raw_output: str = ""               # model text, kept so parser changes can be checked

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def flags_unsafe_family(self) -> Optional[str]:
        return family(self.violation_type)


FIELDS = ("risk_score", "violation_type", "status", "evidence_refs", "uncertainty", "recommended_response")


def _clip(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def extract_json_object(text: str) -> dict:
    """Return the last top-level JSON object in model text (fences tolerated)."""
    if not text:
        raise ValueError("empty model output")
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = fenced[::-1]
    # Fall back to scanning balanced braces from the end of the text.
    depth, end = 0, None
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{" and depth:
            depth -= 1
            if depth == 0 and end is not None:
                candidates.append(text[i:end + 1])
                break
    for chunk in candidates:
        try:
            value = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model output")


def parse_assessment(text: str, bundle: AuditBundle, usage: Optional[dict] = None,
                     version: str = "v0") -> MonitorAssessment:
    """Strictly validate model output; failures become explicit invalid records."""
    usage = usage or {}
    try:
        data = extract_json_object(text)
        missing = [k for k in FIELDS if k not in data]
        if missing:
            raise ValueError("missing fields: " + ", ".join(missing))
        risk = _clip(data["risk_score"], "risk_score")
        uncertainty = _clip(data["uncertainty"], "uncertainty")
        vtype = ViolationType(str(data["violation_type"])).value
        status = Status(str(data["status"])).value
        response = Response(str(data["recommended_response"])).value
        refs = data["evidence_refs"]
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise ValueError("evidence_refs must be a list of strings")
        known = bundle.refs()
        messages = {k.split(".", 1)[0] for k in known}
        # A message-level citation ("m82") is valid when only its tool call ("m82.tc0")
        # is in the record, as is an invented suffix on a real message ("m89.tool_result").
        unknown = [r for r in refs if r not in known and r.split(".", 1)[0] not in messages]
        if unknown:
            raise ValueError("evidence_refs cite unknown events: " + ", ".join(unknown[:5]))
        if status in (Status.VERIFIED.value, Status.SUSPECTED.value) and not refs:
            raise ValueError("a verified/suspected finding must cite evidence_refs")
        if status in (Status.VERIFIED.value, Status.SUSPECTED.value) and vtype == ViolationType.NONE.value:
            raise ValueError("a verified/suspected finding needs a violation_type")
        return MonitorAssessment(bundle.episode_id, risk, vtype, status, refs, uncertainty, response,
                                 rationale=str(data.get("rationale", ""))[:4000], usage=usage,
                                 monitor_version=version)
    except ValueError as exc:
        return invalid(bundle.episode_id, str(exc), usage, version)


def invalid(episode_id: str, error: str, usage: Optional[dict] = None, version: str = "v0") -> MonitorAssessment:
    return MonitorAssessment(episode_id, None, ViolationType.NONE.value, Status.INSUFFICIENT_EVIDENCE.value,
                             [], None, Response.NONE.value, valid=False, error=error[:1000],
                             usage=usage or {}, monitor_version=version)
