"""Download a pinned MisTrace subset to a local Parquet file (needs network + `datasets`).

    python -m loopwatch_monitor.fetch --out data/mistrace-reward-hacking.parquet
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="reward-hacking")
    parser.add_argument("--revision", default=None, help="dataset commit; default = current main, recorded below")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info("ameek/MisTrace").sha
    data = load_dataset("ameek/MisTrace", name=args.subset, split="train", revision=revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(str(args.out))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = {"dataset": "ameek/MisTrace", "subset": args.subset, "revision": revision, "rows": len(data),
            "sha256": digest, "license": "CC-BY-4.0"}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"Pin this in configs/baseline.yaml: benchmark.revision: {revision}")


if __name__ == "__main__":
    main()
