"""Executed inside the container; uses upstream AlgoTuner without a new loop."""
import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "agent", "verify", "reference"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    multiprocessing.set_start_method("forkserver", force=True)
    sys.path.insert(0, "/work/AlgoTune")
    from AlgoTuner.config.loader import load_config
    from AlgoTuneTasks.factory import TaskFactory
    cfg = load_config()
    # Upstream probes filesystem writability beside the oracle source. Route
    # this scratch-only probe elsewhere so pristine source can be read-only.
    # Preserve the original probe implementation and all verifier/timing code.
    from AlgoTuner.utils import isolated_benchmark
    original_health_check = isolated_benchmark._check_filesystem_health
    def health_check(path=None):
        if path and Path(path).resolve().is_relative_to(Path("/work/AlgoTune")):
            path = "/work/health-check"
        return original_health_check(path)
    isolated_benchmark._check_filesystem_health = health_check
    task = TaskFactory(args.task, data_dir="/work/data", target_time_ms=100)
    if args.mode == "prepare":
        data_dir = Path("/work/data") / args.task
        data_dir.mkdir(parents=True, exist_ok=True)
        task.create_dataset(k=args.n, train_size=cfg["dataset"]["train_size"],
                            test_size=cfg["dataset"]["test_size"],
                            random_seed=args.seed, data_dir=str(data_dir))
        return
    if args.mode == "reference":
        # A plumbing check only, never counted as an agent episode.
        task_class = type(task).__name__
        Path("/work/solution/solver.py").write_text(
            f"from AlgoTuneTasks.{args.task}.{args.task} import {task_class}\n"
            f"class Solver:\n    def solve(self, problem, **kwargs):\n        return {task_class}().solve(problem)\n")
    if args.mode in ("verify", "reference"):
        logging.basicConfig(level=logging.INFO, force=True)
        from AlgoTuner.utils.evaluator.baseline_manager import BaselineManager
        from AlgoTuner.utils.evaluator.main import evaluate_code_on_dataset, _calculate_aggregate_metrics
        from scripts.evaluate_results import compile_code_if_needed, needs_compilation
        output = {"completed": False}
        try:
            code_dir = Path("/work/solution")
            if needs_compilation(code_dir)[0]:
                # The upstream build helper otherwise tries to download a fresh
                # build environment. Use the image's installed tools offline.
                # pip's negative boolean option takes 0 to disable isolation.
                os.environ["PIP_NO_BUILD_ISOLATION"] = "0"
                os.environ["PIP_NO_INDEX"] = "1"
                ok, detail = compile_code_if_needed(code_dir, args.task)
                if not ok:
                    raise RuntimeError(detail)
            _, test = task.load_dataset()
            result = evaluate_code_on_dataset(task_obj=task, dataset_iterable=list(test),
                        data_subset="test", baseline_manager=BaselineManager(task))
            if isinstance(result, dict):
                output.update(completed=True, metrics=result.get("aggregate_metrics", {}),
                              reported_failure=result.get("success") is False,
                              upstream_result=result, error=result.get("error"))
            else:
                # Match CommandHandlers._runner_eval_dataset: the low-level
                # result's attached summary contains only accuracy in this pin.
                output.update(completed=True, metrics=_calculate_aggregate_metrics(result),
                              per_problem=[{k: row.get(k) for k in ("problem_id", "success", "is_valid", "error_type",
                                           "baseline_time_ms", "min_time_ms", "solver_min_time_ms", "speedup")} for row in result])
        except Exception as exc:
            output["error"] = str(exc)
        print("LOOPWATCH_PERFORMANCE=" + json.dumps(output, allow_nan=False), flush=True)
        Path("/work/performance.json").write_text(json.dumps(output, allow_nan=False))
        return
    # Instrument only artifact export; preserve upstream initialization/run_task.
    from AlgoTuner.interfaces.llm_interface import LLMInterface
    original = LLMInterface.run_task

    def recorded_run(self):
        try:
            return original(self)
        finally:
            Path("/work/agent_metrics.json").write_text(json.dumps({
                "metrics": getattr(self, "_final_eval_metrics", None),
                "success": getattr(self, "_final_eval_success", None),
                "messages": self.state.messages_sent, "spend": self.state.spend,
            }, default=str))
            Path("/work/agent_history.json").write_text(json.dumps(self.state.messages, default=str))

    LLMInterface.run_task = recorded_run
    from AlgoTuner.main import main as upstream_main
    sys.argv = ["AlgoTuner.main", "--model", "kimi", "--task", args.task]
    upstream_main()


if __name__ == "__main__":
    main()
