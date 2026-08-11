#!/usr/bin/env python3
"""Print the trained LoRA weights .tar URL for a logged training run, needed
by ControlNet+LoRA models (which take raw weights, not a model:version ref)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_LOG = ROOT / "data" / "training_runs.json"


def main():
    if not RUNS_LOG.exists():
        raise SystemExit(f"{RUNS_LOG} not found - no training runs logged yet")

    runs = json.loads(RUNS_LOG.read_text())
    if not runs:
        raise SystemExit("No training runs logged")

    import replicate

    for i, run in enumerate(runs):
        training = replicate.trainings.get(run["id"])
        print(f"Run {i + 1}: id={run['id']} status={training.status}")
        if training.status == "succeeded" and training.output:
            weights = training.output.get("weights") or training.output.get("version")
            print(f"  weights: {weights}")
        print()


if __name__ == "__main__":
    main()
