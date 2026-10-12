#!/usr/bin/env bash
# Runs only inside the disposable preparation VM, before any model episode.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io python3-yaml
systemctl enable --now docker
# Use the checked-out upstream scenario's actual preparation and configuration.
bash /opt/loopwatch/third_party/SandboxEscapeBench/scenarios/docker_socket/setup.sh
docker rm -f eval-container
mkdir -p /opt/build
cp -a /opt/loopwatch/third_party/AlgoTune /opt/build/AlgoTune
cp /opt/loopwatch/runner/Dockerfile /opt/build/Dockerfile
docker build -t loopwatch-algotune /opt/build
mkdir -p /work/solution /work/data
cp -a /opt/build/AlgoTune /work/AlgoTune
cp /opt/loopwatch/configs/algotuner.yaml /work/config.yaml
ln -s AlgoTune/AlgoTuner /work/AlgoTuner
ln -s AlgoTune/AlgoTuneTasks /work/AlgoTuneTasks
ln -s AlgoTune/pyproject.toml /work/pyproject.toml
docker run --rm -v /work:/work -e TIKTOKEN_CACHE_DIR=/work/tokenizer-cache \
    loopwatch-algotune python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); tiktoken.get_encoding("o200k_base")'
docker run --rm --network none -v /work:/work \
    -v /opt/loopwatch/runner/agent.py:/driver/agent.py:ro \
    -e SKIP_DATASET_GEN=0 loopwatch-algotune \
    python /driver/agent.py prepare --task "$1" --n "$2" --seed "$3"
docker run --rm --network none loopwatch-algotune pip freeze >/opt/loopwatch/packages.txt
sync
