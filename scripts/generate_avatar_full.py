#!/usr/bin/env python3
"""One-command end-to-end avatar generation: classical scaffold (structure)
-> Gemini (style) -> InstantID merge (structure-corrected style).

Chains the three separate scripts as subprocesses rather than reimplementing
their logic, so each step stays exactly as tested individually:
    1. composite_line_art.py --scaffold   (accurate jaw/ear/eye structure)
    2. generate_avatar_gemini.py          (polished line style, using the
                                            scaffold as its own structural
                                            anchor)
    3. generate_avatar_instantid.py       (InstantID + ControlNet, seeded
                                            from Gemini's output within the
                                            masked region, corrected back
                                            toward the true structure)

Usage:
    export GEMINI_API_KEY=...
    export REPLICATE_API_TOKEN=...
    python3 scripts/generate_avatar_full.py path/to/photo.jpg --lora-weights https://.../trained_model.tar --glasses
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, step_name):
    print(f"\n=== {step_name} ===")
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"{step_name} failed (exit {result.returncode}) -- stopping pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--lora-weights",
        required=True,
        help="URL to your trained LoRA weights .tar (from scripts/get_lora_weights_url.py)",
    )
    parser.add_argument("--out", help="Where to save the final result. Defaults to <photo>_avatar.png")
    parser.add_argument("--work-dir", help="Where to save intermediate files (scaffold, mask, Gemini draft). Defaults to alongside the photo.")
    parser.add_argument("--glasses", action="store_true", help="subject wears glasses")
    parser.add_argument("--no-glasses", action="store_true", help="subject does not wear glasses")
    parser.add_argument("--beard", action="store_true", help="subject has a beard/visible facial hair")
    parser.add_argument("--no-beard", action="store_true", help="subject is clean-shaven")
    parser.add_argument(
        "--style-denoise",
        type=float,
        default=0.85,
        help="How much the InstantID merge step corrects Gemini's structure/identity and "
        "repaints over its flaws (e.g. soft grey face shading), 0-1. Lower keeps more of "
        "Gemini's actual rendering but also more of those flaws; higher gives the sampler more "
        "room to fully overwrite them while still using Gemini's output as a starting point.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate the scaffold/mask and Gemini draft even if they already exist from a "
        "previous run. Off by default: re-running after only the final merge step failed (e.g. "
        "to retry --style-denoise at a different value) shouldn't have to wait through steps "
        "1-2 again, since neither depends on anything the merge step changes.",
    )
    args = parser.parse_args()

    if args.glasses and args.no_glasses:
        sys.exit("--glasses and --no-glasses are mutually exclusive")
    if args.beard and args.no_beard:
        sys.exit("--beard and --no-beard are mutually exclusive")

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    work_dir = Path(args.work_dir) if args.work_dir else photo_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = photo_path.stem

    scaffold_path = work_dir / f"{stem}_scaffold.png"
    mask_path = work_dir / f"{stem}_scaffold_mask.png"
    gemini_path = work_dir / f"{stem}_gemini.png"
    out_path = Path(args.out) if args.out else work_dir / f"{stem}_avatar.png"

    if not args.force and scaffold_path.exists() and mask_path.exists():
        print(f"\n=== Step 1/3: classical structure scaffold ===\nSkipping -- {scaffold_path} and {mask_path} already exist (pass --force to regenerate)")
    else:
        composite_cmd = [
            sys.executable, str(ROOT / "scripts" / "composite_line_art.py"),
            str(photo_path), str(scaffold_path), "--scaffold",
        ]
        if args.glasses:
            composite_cmd.append("--glasses")
        run(composite_cmd, "Step 1/3: classical structure scaffold")

    if not args.force and gemini_path.exists():
        print(f"\n=== Step 2/3: Gemini style draft ===\nSkipping -- {gemini_path} already exists (pass --force to regenerate)")
    else:
        gemini_cmd = [
            sys.executable, str(ROOT / "scripts" / "generate_avatar_gemini.py"),
            str(photo_path), "--out", str(gemini_path),
        ]
        if args.glasses:
            gemini_cmd.append("--glasses")
        if args.no_glasses:
            gemini_cmd.append("--no-glasses")
        if args.beard:
            gemini_cmd.append("--beard")
        if args.no_beard:
            gemini_cmd.append("--no-beard")
        run(gemini_cmd, "Step 2/3: Gemini style draft")

    instantid_cmd = [
        sys.executable, str(ROOT / "scripts" / "generate_avatar_instantid.py"),
        str(photo_path),
        "--control-image", str(scaffold_path),
        "--mask", str(mask_path),
        "--style-image", str(gemini_path),
        "--style-denoise", str(args.style_denoise),
        "--lora-weights", args.lora_weights,
        "--out", str(out_path),
    ]
    if args.glasses:
        instantid_cmd.append("--glasses")
    if args.no_beard:
        instantid_cmd.append("--no-facial-hair")
    run(instantid_cmd, "Step 3/3: InstantID structure-corrected merge")

    print(f"\nDone. Final avatar: {out_path}")
    print(f"Intermediate files: scaffold={scaffold_path}, mask={mask_path}, gemini_draft={gemini_path}")


if __name__ == "__main__":
    main()
