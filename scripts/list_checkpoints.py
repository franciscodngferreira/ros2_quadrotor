#!/usr/bin/env python3
"""Print checkpoint files and training_summary.json from a training run."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.dir)
    if not os.path.isdir(ckpt_dir):
        print(f"No directory: {ckpt_dir}")
        return

    summary_path = os.path.join(ckpt_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        print("=== training_summary.json ===")
        print(json.dumps(summary, indent=2))
        print()

    zips = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".zip"))
    if not zips:
        print(f"No .zip checkpoints in {ckpt_dir}")
        return

    print(f"=== checkpoints in {ckpt_dir} ===")
    for name in zips:
        path = os.path.join(ckpt_dir, name)
        size_kb = os.path.getsize(path) / 1024
        meta_path = path.replace(".zip", "_meta.json")
        extra = ""
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            extra = f"  reward={meta.get('mean_reward', '?')}  steps={meta.get('timesteps', '?')}"
        print(f"  {name}  ({size_kb:.0f} KB){extra}")

    print()
    print("Recommended for eval: checkpoints/best_eval.zip")


if __name__ == "__main__":
    main()
