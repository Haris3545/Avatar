#!/usr/bin/env python3
"""Fuzzy-match avatars to headshots by filename, to bootstrap the training pair dataset."""
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVATAR_DIR = ROOT / "data" / "avatars"
HEADSHOT_DIR = ROOT / "data" / "headshots"
OUT_CSV = ROOT / "data" / "manifest.csv"

NOISE = re.compile(
    r"\b(avatar|photo|headshot|linkedin|hi|sq|col|mono|copy|greene king|gch|vccp media|\d{4}|\(\d+\)|v\d+|-\s*\d+)\b",
    re.IGNORECASE,
)


def normalize(stem: str) -> str:
    s = NOISE.sub(" ", stem)
    s = re.sub(r"[^a-z\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def list_files(d: Path):
    return [f for f in d.iterdir() if f.is_file() and f.name != ".gitkeep"]


def main():
    avatars = list_files(AVATAR_DIR)
    headshots = list_files(HEADSHOT_DIR)

    avatar_keys = {f: normalize(f.stem) for f in avatars}
    headshot_keys = {f: normalize(f.stem) for f in headshots}

    rows = []
    used_headshots = set()
    for av, av_key in sorted(avatar_keys.items(), key=lambda kv: kv[0].name):
        best, best_score = None, 0.0
        for hs, hs_key in headshot_keys.items():
            if hs in used_headshots:
                continue
            s = score(av_key, hs_key)
            if s > best_score:
                best, best_score = hs, s
        if best and best_score >= 0.5:
            used_headshots.add(best)
            rows.append((av.name, best.name, round(best_score, 2)))
        else:
            rows.append((av.name, "", round(best_score, 2)))

    unmatched_headshots = [h.name for h in headshots if h not in used_headshots]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["avatar", "headshot", "confidence"])
        w.writerows(rows)
        for h in unmatched_headshots:
            w.writerow(["", h, ""])

    confident = [r for r in rows if r[1] and r[2] >= 0.75]
    review = [r for r in rows if r[1] and r[2] < 0.75]
    unmatched_avatars = [r for r in rows if not r[1]]

    print(f"Avatars: {len(avatars)}  Headshots: {len(headshots)}")
    print(f"Confident matches (>=0.75): {len(confident)}")
    print(f"Needs review (0.5-0.75): {len(review)}")
    print(f"Unmatched avatars: {len(unmatched_avatars)}")
    print(f"Unmatched headshots: {len(unmatched_headshots)}")
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
