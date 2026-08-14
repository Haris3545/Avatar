#!/usr/bin/env python3
"""One-time (re-runnable, cached) classification pass over data/avatars/:
for each reference avatar, ask a Gemini vision model whether the subject
is drawn wearing glasses and/or with a beard/visible facial hair, and
cache the result to data/avatar_tags.json.

generate_avatar_gemini.py uses this cache to filter which reference
avatars get sent for a given subject (e.g. only send glasses-wearing
references for a subject who wears glasses) instead of sending the whole
set regardless of fit. Sending the unfiltered set produced a result that
was more polished than the house style but didn't actually look like it --
averaging in-context examples across many different feature combinations
(glasses vs not, beard vs not) muddies exactly the per-feature drawing
conventions (how glasses are drawn, how a beard is drawn) that make an
avatar read as "ours" for a specific class of subject.

Usage:
    export GEMINI_API_KEY=...
    python3 scripts/tag_avatar_references.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVATARS_DIR = ROOT / "data" / "avatars"
TAGS_PATH = ROOT / "data" / "avatar_tags.json"

PROMPT = (
    "Look at this illustrated avatar portrait. Answer with strict JSON only, no other text, "
    'in exactly this shape: {"glasses": true|false, "beard": true|false}. '
    '"glasses" is true only if the subject is drawn wearing glasses. '
    '"beard" is true only if the subject has an actual visible beard or moustache covering the '
    "cheeks, chin, and/or upper lip as real facial hair. "
    "Do NOT count a small confined dot-pattern shading patch directly under the chin as a beard -- "
    "that's a stylized jaw-shadow/5-o'clock-shadow indicator used on many avatars regardless of "
    "whether the subject has real facial hair, not an actual beard. Only a clearly drawn beard "
    "shape (covering a wider area, usually including the sides of the face and/or moustache) counts."
)


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-classify everything, overwriting the existing cache (e.g. after changing PROMPT)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey)")

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=api_key)

    avatars = sorted(
        p for p in AVATARS_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    tags = {}
    if TAGS_PATH.exists() and not args.force:
        tags = json.loads(TAGS_PATH.read_text())

    import time

    for p in avatars:
        if p.name in tags:
            continue
        print(f"Classifying {p.name}...")

        response = None
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=[PROMPT, Image.open(p)],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                break
            except Exception as e:
                # The API returns transient 503s under load -- retry with
                # backoff instead of failing the whole batch over one flaky
                # response, especially since each successful call is billed.
                if attempt == 3:
                    print(f"  giving up on {p.name} after retries: {e}")
                else:
                    wait = 2**attempt * 3
                    print(f"  transient error ({e.__class__.__name__}), retrying in {wait}s...")
                    time.sleep(wait)
        if response is None:
            continue

        try:
            result = json.loads(response.text)
            tags[p.name] = {"glasses": bool(result["glasses"]), "beard": bool(result["beard"])}
            print(f"  -> {tags[p.name]}")
        except Exception as e:
            print(f"  failed to parse response for {p.name}: {e}\n  raw: {response.text!r}")
            continue
        # Write incrementally so an interrupted run doesn't lose progress
        # already paid for.
        TAGS_PATH.write_text(json.dumps(tags, indent=2, sort_keys=True))

    print(f"Tagged {len(tags)}/{len(avatars)} avatars -> {TAGS_PATH}")


if __name__ == "__main__":
    main()
