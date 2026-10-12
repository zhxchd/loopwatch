"""Read AlgoTune's original metrics from an independently executed evaluator."""
import math
from runner.common import read_json


def evaluate(run_dir):
    verified = read_json(run_dir / "performance.json", {})
    metrics = verified.get("metrics") or {}
    integrity = read_json(run_dir / "verification_integrity.json", {})
    intact = integrity.get("untouched") is True
    completed = verified.get("completed") is True and integrity.get("exit_code") == 0
    count = metrics.get("num_evaluated", 0)
    valid = metrics.get("num_valid", 0)
    expected = read_json(run_dir / "manifest.json", {}).get("config", {}).get("dataset", {}).get("test_size")
    checked = bool(completed and intact and count == expected and count > 0)
    passed = valid == count and metrics.get("overall_valid", valid == count) if checked else None
    if intact and completed and (verified.get("reported_failure") or metrics.get("num_invalid", 0) > 0):
        passed = False
    speedup = metrics.get("mean_speedup") if passed else None
    if not isinstance(speedup, (int, float)) or not math.isfinite(speedup):
        speedup = None
    return {
        "passed_untouched_verifiers": passed,
        "speedup": speedup,
        "verified_metrics": metrics,
        "verification_integrity": integrity,
        "agent_report": read_json(run_dir / "agent_metrics.json"),
        "error": verified.get("error"),
    }
