#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--label", action="append", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.source)
    target_path = Path(args.target)

    with source_path.open() as handle:
        source = json.load(handle)
    with target_path.open() as handle:
        target = json.load(handle)

    source_runs = source.get("runs", {})
    target.setdefault("runs", {})
    target.setdefault("comparisons", {})
    target.setdefault("comparison_baselines", {})

    for label in args.label:
        if label not in source_runs:
            raise ValueError(f"Source shard does not contain run '{label}'.")
        target["runs"][label] = source_runs[label]
        if label in source.get("comparisons", {}):
            target["comparisons"][label] = source["comparisons"][label]
        if label in source.get("comparison_baselines", {}):
            target["comparison_baselines"][label] = source["comparison_baselines"][label]

    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with temp_path.open("w") as handle:
        json.dump(target, handle, indent=2)
    temp_path.replace(target_path)


if __name__ == "__main__":
    main()
