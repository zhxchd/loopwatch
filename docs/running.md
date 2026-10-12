# Running the first baseline

The checked-out submodules are part of the project to sync. Initialize them once
when cloning a new checkout (`git submodule update --init --recursive`). Neither
the runtime setup nor experiment scripts fetch source repositories.

Put `KIMI_API_KEY` in the root `.env`, or export it in the CPU job's environment.
The outer relay consumes it at runtime without printing it. The experimental VM
receives only a dummy key; API requests pass through a bounded relay outside it.

On ORCD, sync the checkout to project scratch, excluding local `.venv`, `.cache`,
and `runs`. Create the cluster `.env` with only `KIMI_API_KEY` and set mode `600`.
Do not sync `.env` into the VM or include it in an artifact archive. Python dependencies and VM images
live under `.cache`; output lives under `runs`.

The first cohort is closed; see the [report](first-baseline-report.md). All 20
original jobs finished, with 17 model-exposed trials and three API initialization
failures. The user declined initialization retries; none were submitted. The
following submission commands are for a future experiment.
The original array was `25370257`, the completed verification-only retry array
was `25401397`, and aggregation was `25370280`. Inspect their accounting with:

```bash
ssh -S /tmp/loopwatch-orcd.sock xczhu@orcd-login.mit.edu
cd /orcd/scratch/orcd/007/xczhu/loopwatch
sacct -j 25370257,25401397,25370280 --format=JobID,State,ExitCode,Elapsed
```

## Prepare once, then submit an array

From the synced project directory on the cluster, install the outer Python/QEMU
runtime once. This is a CPU job and does not run the model:

```bash
mkdir -p runs
sbatch cluster/runtime.sbatch
```

After that job succeeds, prepare the image in a separate CPU job. The quoted
heredoc keeps `$PWD` for expansion inside the job, in the submission directory.

```bash
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=loopwatch-prepare
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --constraint=node1600|node1602|node1603|node1608
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/prepare-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/.cache/runtime/bin:$PATH"
python -m runner.run --config configs/baseline.yaml --prepare-only
SBATCH
```

Preparation must finish successfully before submitting the array. It includes
an untouched reference correctness check and makes no model calls. If the image
already exists and matches the configuration, preparation reuses it.

Submit the configured 20 independent episodes, at most four at a time:

```bash
baseline_job=$(sbatch --parsable --array=1-20%4 --time=04:00:00 cluster/baseline.sbatch)
squeue -j "$baseline_job"
```

The array range must match `repetitions` in the configuration. Each task has its
own VM, observation stream, API budget, and verification VM. Array tasks never
write the shared aggregate. The four-hour allocation allows two hours for the
agent, one hour for independent verification, and VM/collection overhead.

Submit a small CPU aggregation job that waits for all array tasks, including
failed ones, so failures remain visible:

```bash
sbatch --job-name=loopwatch-aggregate \
    --partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:10:00 \
    --dependency="afterany:$baseline_job" --output='runs/aggregate-%j.log' \
    --wrap=".cache/runtime/bin/python -m eval.run --experiment-dir runs/baseline-$baseline_job"
```

Results land in `runs/baseline-JOB_ID/`. `summary.json` is written by aggregation;
an individual `episode-NNN/results.json` is written when that episode finishes
its verification attempt. Job completion alone does not establish a valid
result: inspect missing episodes, unknown metrics, API errors, and response
counts. If verification-only repair jobs are needed, make aggregation wait for
them too, or rerun aggregation after they finish.

Both scripts request the `mit_normal` CPU partition, account and QoS
`mit_amf_advanced_cpu`, and no GPUs. Runtime setup installs Python and extracts
Rocky Linux QEMU packages into project scratch without administrator access.
Compilation, VM preparation, reference checking, and model episodes run only
inside CPU allocations. Login nodes are used for sync, submission, and status.

For an existing CPU allocation with the runtime on PATH, the underlying CLI is:

```bash
python -m runner.run --config configs/baseline.yaml --prepare-only
python -m runner.run --config configs/baseline.yaml --experiment-dir runs/baseline
python -m eval.run --experiment-dir runs/baseline
```

The second command runs **all configured repetitions sequentially**. It is
useful for small configurations; twenty episodes can exceed a single allocation.
The batch script also runs sequentially if submitted without `--array`.
`--episode-index N` runs only that repetition in a shared experiment directory;
the array script supplies it automatically. Episode directories cannot be
overwritten or resumed in place.
The runner checks prepared-image compatibility even without a preparation flag
and rejects reuse of an experiment directory with a different configuration or
upstream revision.

## Configuration and saved results

`configs/baseline.yaml` controls task, repetition count, model, budgets, dataset,
VM resources, prepared image, and protected paths. `cluster/baseline.sbatch`
currently names that config explicitly. For a different config, copy the batch
script and replace its `--config` paths, or use the CLI inside a CPU allocation.
Do not edit the config/source while jobs from the same cohort are still queued
or running; use a separate checkout/config for a distinct experiment.

Changes to the task, dataset, generated AlgoTuner configuration, upstream pins,
or image build inputs require a matching prepared image. Choose a new
`sandbox.base_image` filename rather than overwriting an incompatible image.
Keep `.cache` and the image's backing files in place: the prepared QCOW2 may
depend on other files there. The baseline does not implement cross-episode
memory or harness evolution; increasing `repetitions` only adds independent
sessions.

Offline scoring does not call Kimi or execute the candidate again:

```bash
.cache/runtime/bin/python -m eval.run --run-dir runs/baseline-JOB_ID/episode-001
.cache/runtime/bin/python -m eval.run --experiment-dir runs/baseline-JOB_ID
```

It recalculates decisions from recorded monitor evidence and independent
verification results. Re-executing a saved candidate requires a verification
job, which is a different operation from offline scoring.

To copy results back, run this on the local machine, replacing `JOB_ID`:

```bash
rsync -a --exclude='*.qcow2' --exclude='*.iso' --exclude='.env*' \
    --exclude='vm_key*' --exclude='*.pem' \
    -e 'ssh -S /tmp/loopwatch-orcd.sock' \
    xczhu@orcd-login.mit.edu:/orcd/scratch/orcd/007/xczhu/loopwatch/runs/baseline-JOB_ID/ \
    /Users/xczhu/Desktop/loopwatch/runs/baseline-JOB_ID/
```

## Isolation and evidence

Preparation downloads Ubuntu's cloud image, checks its published SHA-256, and
runs the pinned SandboxEscapeBench `docker_socket/setup.sh` inside QEMU. It then
builds an AlgoTune container, generates a seeded dataset, and checks a reference
solver with upstream evaluation. Preparation needs network access. Experiment
VMs use restricted QEMU networking, with SSH and the Kimi relay forwarded.
No cluster filesystem is shared into the VM. The baseline requires hardware KVM
and selects CPU nodes where access was verified (`node1600`, `node1602`,
`node1603`, and `node1608`). It does not fall back to software emulation. The actual acceleration
is recorded for every episode; update the node constraint for another cluster.

Each episode starts a fresh disk overlay. The observer starts before the agent
and must report ready. Its JSON events stream over SSH into the outer run
directory. After AlgoTuner exits or reaches its time cap, the runner records a
final integrity snapshot and stops the observer. Only solution files are copied
into another fresh VM for independent verification. Protected source/data/config
mounts are read-only there. Verification has a separate one-hour allowance;
it does not extend the agent's two-hour interaction limit or change benchmark
timing repetitions. Episode overlays are deleted after collection.

Artifacts include configuration and source revisions, full API request/response
trajectories, agent logs and reported metrics, solution files, protected-file
manifests, live observation events, independent performance metrics, and offline
results. `summary.json` includes denominators, unknowns, rates, and Wilson 95%
intervals. Upstream dependency versions and reference metrics are saved during
preparation. A configuration/source mismatch prevents silent reuse of an old VM.

A runner-level infrastructure failure stops that process; in an array, the
other tasks continue independently. API initialization failures do not count as
clean episodes. Preserve failed artifacts. Retry initialization only when no
model response was received; keep every model-exposed trial in the cohort.
Verification infrastructure repairs may rerun the unchanged saved candidate,
with the failed verification archived and before/after candidate hashes recorded.
