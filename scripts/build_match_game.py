#!/usr/bin/env python3
"""Build a self-contained HTML matching game for pairing leftover avatars/headshots
that the filename-based matcher couldn't confidently pair."""
import base64
import csv
import io
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AVATAR_DIR = ROOT / "data" / "avatars"
HEADSHOT_DIR = ROOT / "data" / "headshots"
MANIFEST_CSV = ROOT / "data" / "manifest.csv"
OUT_HTML = ROOT / "scripts" / "match_game.html"

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


def thumb_data_uri(path: Path, size=160) -> str | None:
    try:
        im = Image.open(path)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def main():
    confident_avatars, confident_headshots = set(), set()
    with open(MANIFEST_CSV) as f:
        for row in csv.DictReader(f):
            if row["avatar"] and row["headshot"] and row["confidence"] and float(row["confidence"]) >= 0.75:
                confident_avatars.add(row["avatar"])
                confident_headshots.add(row["headshot"])

    matched_person_keys = {normalize(Path(n).stem) for n in confident_avatars}

    all_avatars = sorted(f for f in AVATAR_DIR.iterdir() if f.is_file() and f.name != ".gitkeep")
    all_headshots = sorted(f for f in HEADSHOT_DIR.iterdir() if f.is_file() and f.name != ".gitkeep")

    candidate_avatars = [f for f in all_avatars if f.name not in confident_avatars]

    candidate_headshots, extras = [], []
    for f in all_headshots:
        if f.name in confident_headshots:
            continue
        key = normalize(f.stem)
        is_extra = any(score(key, mk) >= 0.6 for mk in matched_person_keys)
        (extras if is_extra else candidate_headshots).append(f)

    avatar_items, headshot_items = [], []
    for f in candidate_avatars:
        uri = thumb_data_uri(f)
        if uri:
            avatar_items.append({"name": f.name, "src": uri})
    for f in candidate_headshots:
        uri = thumb_data_uri(f)
        if uri:
            headshot_items.append({"name": f.name, "src": uri})
        else:
            extras.append(f)  # unreadable format (e.g. jxl) -> can't show, treat as skipped

    skipped = [f.name for f in extras]

    data = {"avatars": avatar_items, "headshots": headshot_items, "skipped": skipped}
    OUT_HTML.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)))
    print(f"Avatars in game: {len(avatar_items)}")
    print(f"Headshots in game: {len(headshot_items)}")
    print(f"Skipped (duplicate/unreadable): {len(skipped)}")
    print(f"Written to {OUT_HTML}")


TEMPLATE = """<title>Pairing Bench &mdash; VCCP Avatars</title>
<style>
  :root {
    --ground: #f4f3ef;
    --surface: #ffffff;
    --surface-sunken: #ece9e2;
    --ink: #1f1d18;
    --ink-muted: #6f6a5e;
    --line: #ddd8cc;
    --accent: #a8461f;
    --accent-ink: #fff8f2;
    --focus: #1f1d18;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #17160f;
      --surface: #221f17;
      --surface-sunken: #1c1a13;
      --ink: #f2ede1;
      --ink-muted: #a49b87;
      --line: #3a362a;
      --accent: #e08c5b;
      --accent-ink: #1c1108;
      --focus: #f2ede1;
    }
  }
  :root[data-theme="dark"] {
    --ground: #17160f;
    --surface: #221f17;
    --surface-sunken: #1c1a13;
    --ink: #f2ede1;
    --ink-muted: #a49b87;
    --line: #3a362a;
    --accent: #e08c5b;
    --accent-ink: #1c1108;
    --focus: #f2ede1;
  }

  * { box-sizing: border-box; }
  body {
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--ground);
    color: var(--ink);
    margin: 0;
    padding: 32px 28px 60px;
  }
  header { max-width: 1180px; margin: 0 auto 28px; }
  .eyebrow {
    font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    font-size: 11px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 8px;
  }
  h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; text-wrap: balance; }
  p.sub { color: var(--ink-muted); margin: 0; font-size: 14px; max-width: 60ch; line-height: 1.5; }

  .board { max-width: 1180px; margin: 0 auto; display: flex; gap: 22px; align-items: flex-start; }
  .col { flex: 1; min-width: 0; }
  .col-head {
    display: flex; align-items: baseline; justify-content: space-between;
    font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-muted);
    margin: 0 0 10px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
  }
  .col-head .count { font-variant-numeric: tabular-nums; color: var(--ink); }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap: 10px; }
  .card {
    cursor: pointer; border: 1.5px solid var(--line); border-radius: 10px; padding: 7px;
    text-align: center; background: var(--surface); transition: border-color .12s, transform .12s;
  }
  .card:hover { border-color: var(--ink-muted); }
  .card:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  .card img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; display: block; background: var(--surface-sunken); }
  .card .name { font-size: 10.5px; margin-top: 6px; word-break: break-word; line-height: 1.25; color: var(--ink-muted); }
  .card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .empty { color: var(--ink-muted); font-size: 13px; padding: 28px 8px; text-align: center; border: 1px dashed var(--line); border-radius: 10px; }

  .pairs { max-width: 1180px; margin: 34px auto 0; }
  .pairs table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .pairs th {
    text-align: left; padding: 6px 10px; font-family: ui-monospace, monospace; font-size: 10.5px;
    text-transform: uppercase; letter-spacing: .06em; color: var(--ink-muted); border-bottom: 1px solid var(--line);
  }
  .pairs td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  .pairs tbody tr:hover { background: var(--surface); }
  .pairs a { color: var(--accent); text-decoration: none; font-size: 12px; }
  .pairs a:hover { text-decoration: underline; }

  .actions { display: flex; gap: 10px; margin-top: 16px; }
  button {
    font: inherit; font-size: 13px; font-weight: 600; border: 1px solid transparent;
    padding: 9px 16px; border-radius: 8px; cursor: pointer;
  }
  button#exportBtn { background: var(--accent); color: var(--accent-ink); }
  button#exportBtn:hover { opacity: .9; }
  button.secondary { background: transparent; border-color: var(--line); color: var(--ink); }
  button.secondary:hover { border-color: var(--ink-muted); }
</style>
<header>
  <p class="eyebrow">Training pair dataset &middot; step 2 of 2</p>
  <h1>Pairing bench</h1>
  <p class="sub">These couldn't be paired by filename. Click a face on the left, then its match on the right &mdash; if an avatar genuinely has no headshot here, select it and mark it as unmatched. Export when done and send the CSV back.</p>
</header>
<div class="board">
  <div class="col">
    <div class="col-head"><span>Avatars</span><span class="count" id="avatarCount"></span></div>
    <div class="grid" id="avatarGrid"></div>
  </div>
  <div class="col">
    <div class="col-head"><span>Headshots</span><span class="count" id="headshotCount"></span></div>
    <div class="grid" id="headshotGrid"></div>
  </div>
</div>
<div class="pairs">
  <div class="col-head"><span>Paired so far</span></div>
  <table id="pairTable"><thead><tr><th>Avatar</th><th>Headshot</th><th></th></tr></thead><tbody></tbody></table>
  <div class="actions">
    <button id="exportBtn">Export pairs_manual.csv</button>
    <button class="secondary" id="skipAvatarBtn">Mark selected avatar as "no headshot"</button>
  </div>
</div>
<script>
const DATA = __DATA__;
let selectedAvatar = null;
const pairs = []; // {avatar, headshot}
const noMatch = []; // avatar names with no headshot

function render() {
  const aGrid = document.getElementById('avatarGrid');
  const hGrid = document.getElementById('headshotGrid');
  aGrid.innerHTML = '';
  hGrid.innerHTML = '';

  const pairedAvatars = new Set(pairs.map(p => p.avatar).concat(noMatch));
  const pairedHeadshots = new Set(pairs.map(p => p.headshot));

  if (DATA.avatars.length === 0) aGrid.innerHTML = '<div class="empty">No avatars left</div>';
  if (DATA.headshots.length === 0) hGrid.innerHTML = '<div class="empty">No headshots left</div>';

  let avatarsLeft = 0, headshotsLeft = 0;

  DATA.avatars.forEach(item => {
    if (pairedAvatars.has(item.name)) return;
    avatarsLeft++;
    const el = document.createElement('div');
    el.className = 'card' + (selectedAvatar === item.name ? ' selected' : '');
    el.tabIndex = 0;
    el.innerHTML = `<img src="${item.src}" alt=""><div class="name">${item.name}</div>`;
    const pick = () => { selectedAvatar = (selectedAvatar === item.name) ? null : item.name; render(); };
    el.onclick = pick;
    el.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
    aGrid.appendChild(el);
  });

  DATA.headshots.forEach(item => {
    if (pairedHeadshots.has(item.name)) return;
    headshotsLeft++;
    const el = document.createElement('div');
    el.className = 'card';
    el.tabIndex = 0;
    el.innerHTML = `<img src="${item.src}" alt=""><div class="name">${item.name}</div>`;
    const pick = () => {
      if (!selectedAvatar) return;
      pairs.push({ avatar: selectedAvatar, headshot: item.name });
      selectedAvatar = null;
      render();
    };
    el.onclick = pick;
    el.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
    hGrid.appendChild(el);
  });

  document.getElementById('avatarCount').textContent = avatarsLeft;
  document.getElementById('headshotCount').textContent = headshotsLeft;

  const tbody = document.querySelector('#pairTable tbody');
  tbody.innerHTML = '';
  pairs.forEach((p, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${p.avatar}</td><td>${p.headshot}</td><td><a href="#" data-i="${i}">remove</a></td>`;
    tr.querySelector('a').onclick = (e) => { e.preventDefault(); pairs.splice(i, 1); render(); };
    tbody.appendChild(tr);
  });
  noMatch.forEach((name, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${name}</td><td><em>no headshot</em></td><td><a href="#" data-i="${i}">undo</a></td>`;
    tr.querySelector('a').onclick = (e) => { e.preventDefault(); noMatch.splice(i, 1); render(); };
    tbody.appendChild(tr);
  });
}

document.getElementById('skipAvatarBtn').onclick = () => {
  if (!selectedAvatar) { alert('Select an avatar first'); return; }
  noMatch.push(selectedAvatar);
  selectedAvatar = null;
  render();
};

document.getElementById('exportBtn').onclick = () => {
  let csv = 'avatar,headshot\\n';
  pairs.forEach(p => csv += `"${p.avatar}","${p.headshot}"\\n`);
  noMatch.forEach(n => csv += `"${n}",\\n`);
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pairs_manual.csv';
  a.click();
};

render();
</script>
"""

if __name__ == "__main__":
    main()
