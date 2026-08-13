#!/usr/bin/env python3
"""Kick off an SDXL LoRA training run on Replicate, capped at MAX_RUNS total
to keep spend predictable (~$0.60-1 per run).

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/train_lora.py --destination yourusername/vccp-avatar-lora
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ZIP = ROOT / "data" / "lora_dataset.zip"
RUNS_LOG = ROOT / "data" / "training_runs.json"
MAX_RUNS = 10  # raised from 5 -- explicitly authorized after adding more budget (~£5)

DEFAULT_PARAMS = {
    "input_images": None,  # filled in with an uploaded file URL at runtime
    "token_string": "TOK",
    "caption_prefix": "",
    "use_face_detection_instead": False,
    "max_train_steps": 1500,
    "is_lora": True,
    "lora_lr": 1e-4,
    "resolution": 1024,
}


def load_runs():
    if RUNS_LOG.exists():
        return json.loads(RUNS_LOG.read_text())
    return []


def save_run(record):
    runs = load_runs()
    runs.append(record)
    RUNS_LOG.write_text(json.dumps(runs, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, help="e.g. yourusername/vccp-avatar-lora")
    parser.add_argument("--steps", type=int, default=DEFAULT_PARAMS["max_train_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULT_PARAMS["lora_lr"])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    runs = load_runs()
    if len(runs) >= MAX_RUNS:
        raise SystemExit(
            f"Training run cap reached ({MAX_RUNS} runs already logged in {RUNS_LOG}). "
            "Delete/edit that file if you deliberately want to run again (extra spend)."
        )

    if not DATASET_ZIP.exists():
        raise SystemExit(f"{DATASET_ZIP} not found - run scripts/prepare_lora_dataset.py first")

    import replicate  # imported here so the script still runs --help without the package installed

    print(f"Run {len(runs) + 1}/{MAX_RUNS} - uploading dataset...")
    with open(DATASET_ZIP, "rb") as f:
        uploaded = replicate.files.create(f)

    training = replicate.trainings.create(
        model="stability-ai/sdxl",
        version="7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "input_images": uploaded.urls["get"],
            "input_images_filetype": "zip",
            # No "autocaption" field: cog-sdxl's actual schema has no such
            # parameter (confirmed against its source) -- it was silently
            # ignored on every past run. Captioning is decided purely by
            # whether captions.csv exists in the zip (see
            # prepare_lora_dataset.py); if it's missing, this falls back to
            # BLIP auto-captioning with no warning, which is what was
            # actually happening despite this line's old comment.
            "token_string": DEFAULT_PARAMS["token_string"],
            "max_train_steps": args.steps,
            "is_lora": True,
            "lora_lr": args.lr,
            "resolution": DEFAULT_PARAMS["resolution"],
            "crop_based_on_salience": False,  # dataset is already uniformly cropped/composited
        },
        destination=args.destination,
    )

    save_run(
        {
            "id": training.id,
            "destination": args.destination,
            "steps": args.steps,
            "lr": args.lr,
            "notes": args.notes,
        }
    )

    print(f"Training started: {training.id}")
    print(f"Track at: https://replicate.com/p/{training.id}")


if __name__ == "__main__":
    main()
