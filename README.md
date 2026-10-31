# loopwatch

**Updates.** We moved to ImpossibleBench, where some results are for now available in https://github.com/zhxchd/impossiblebench/tree/selfimprove-level2. Monitoring results are available in this repo in the  at [monitoring branch](https://github.com/zhxchd/loopwatch/tree/monitoring).

A small baseline for unprompted unsafe behavior during algorithm optimization.
Each repetition starts the upstream **AlgoTuner** on the same fixed AlgoTune task
in a fresh VM. Episodes have no shared memory or self-modifying outer harness.

The first configuration uses `count_connected_components`, Kimi K2.6, and 20
independent episodes. One episode is a normal multi-command AlgoTuner session,
bounded by 20 commands, two hours, and a conservative $1 API allowance. The wall
clock allowance includes upstream's substantial isolated-evaluation overhead.

```text
configs/       experiment parameters
runner/        AlgoTuner adapter, VM runner, API relay, live observer
eval/          independent performance, protected-file, and escape scoring
cluster/       CPU-only ORCD submission scripts
docs/          execution and measurement notes
third_party/   pinned AlgoTune and SandboxEscapeBench submodules
runs/          local experiment artifacts (ignored by Git)
```

See [running](docs/running.md) and [evaluation](docs/evaluation.md).

The [first baseline report](docs/first-baseline-report.md) records 17 model-exposed
trials from 20 submitted slots; three received no model response during a Kimi
overload and were not retried. There were no measured unsafe incidents (0/17;
Wilson 95% interval 0–18.4%). All 17 passed untouched verification, with a median
speedup of 8.27×. The report documents shortened runs and measurement limitations.
