# ORCD runbook

How to run experiments on MIT ORCD Engaging.

## Reaching the cluster

The login node is `orcd-login.mit.edu` and the user is `xczhu`.

The user has authenticated via Kerberos and Duo and running from a user's terminal works without any other setup and is automatically authenticated via ssh keys. Running from an agent sandbox may fail, so do not use that.

## Cluster facts

- Accounts (premium):
  - GPU jobs: `--partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu`
  - CPU jobs: `--partition=mit_normal --account=mit_amf_advanced_cpu --qos=mit_amf_advanced_cpu`
- The GPU QoS allows each user 4 GPUs, 64 CPUs and 1 TB at a time. Its 2-day
  wall limit overrides the partition's 6 hours. So at most two 2-GPU jobs run at
  once; further jobs pend with reason `QOSMaxGRESPerUser`.
- `mit_normal_gpu` holds mostly L40S nodes (4 GPUs each), plus H200 nodes
  (8 each) and one H100 node.
- Compute nodes have internet access (Hugging Face, PyPI, Docker Hub), NVIDIA
  driver 590 (CUDA 13.1), and a local `/tmp`.
  - On compute nodes, `apptainer` is available only through
    `module load apptainer/1.4.2`. The login node has `/usr/bin/apptainer` for
    quick checks.
  - Inside containers, home is bound automatically. Scratch must be bound by its
    real path: `readlink -f ~/orcd/scratch`.
- Storage quotas: home 200 GB (backed up); scratch 1 TB (flash, deleted only
  after 6 months without a login); pool 1 TB (HDD, unused). Check usage with
  `cat ~/orcd/.quota`.
- Use the login node only for light work: `sbatch`, `squeue`, small checks.
