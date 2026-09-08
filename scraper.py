"""Scrape AMC 10 problems from live.poshenloh.com (MAA-licensed, LaTeX source).

Dataset slicing rules (per contest):
  - contests before 2010:            last  5 problems
  - contests from 2010 through 2019: last 10 problems
  - contests from 2020 through 2025: last 15 problems

Outputs:
  data/problems.jsonl  - one problem per line with metadata
  data/corpus.txt      - char-level training corpus
"""

import json
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://live.poshenloh.com/past-contests/amc10"

# Contest slugs: 2000/2001 had a single contest; 2021 has four (A/B spring, C/D fall).
CONTESTS = ["2000", "2001"]
for year in range(2002, 2026):
    if year == 2021:
        CONTESTS += ["2021A", "2021B", "2021C", "2021D"]
    else:
        CONTESTS += [f"{year}A", f"{year}B"]


def n_take(year: int) -> int:
    """How many trailing problems to keep for a given contest year."""
    if year < 2010:
        return 5
    if year < 2020:
        return 10
    return 15  # 2020-2025


def year_of(slug: str) -> int:
    return int(slug[:4])


def label_of(slug: str) -> str:
    if slug in ("2021C", "2021D"):
        return "Fall" + slug[4]
    return slug[4] if len(slug) > 4 else "S"


def fetch(slug: str) -> str:
    url = f"{BASE}/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


IMG_RE = re.compile(r"<img[^>]*>")


def clean(text: str) -> str:
    """Remove figure tags; collapse excess blank lines."""
    text = IMG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_next_data(html: str) -> list[dict]:
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        return []
    data = json.loads(m.group(1))
    return data["props"]["pageProps"]["baseQuestions"]


def latex_choice(v: str) -> str:
    v = v.strip()
    if v.startswith("\\(") and v.endswith("\\)"):
        v = v[2:-2].strip()
    return v


def format_problem(year: int, season: str, num: int, text: str, choices: dict[str, str]) -> str:
    lines = [f"YEAR: {year}", f"CONTEST: {season}", f"PROBLEM: {num}", "", text]
    ordered = [(L, choices[L]) for L in "ABCDE" if L in choices]
    if ordered:
        lines.append("")
        lines.append(" ".join(f"({L}) {v}" for L, v in ordered))
    return "\n".join(lines)


def main() -> None:
    out = Path("data")
    out.mkdir(exist_ok=True)
    records = []
    for slug in CONTESTS:
        year = year_of(slug)
        take = n_take(year)
        try:
            html = fetch(slug)
        except Exception as e:
            print(f"!! {slug}: {e}")
            continue
        try:
            qs = parse_next_data(html)
        except Exception as e:
            print(f"!! {slug}: parse error {e}")
            continue
        if not qs:
            print(f"!! {slug}: no baseQuestions found")
            continue

        # Array position is the true problem order (the site's amc10ProblemNumber
        # field is unreliable for early years). Index 0 == Problem 1.
        # Keep the LAST `take` problems of the contest.
        kept = list(enumerate(qs))[-take:]
        season = label_of(slug)

        for i, q in kept:
            n = i + 1
            text = clean(q.get("question", ""))
            choices = {}
            for L in "ABCDE":
                v = q.get(L.lower())
                if isinstance(v, str) and v.strip():
                    choices[L] = latex_choice(v)
            correct = (q.get("answer") or "").strip().upper()
            if not text or len(choices) < 5 or correct not in "ABCDE":
                print(f"  skip {slug} #{n}: incomplete")
                continue
            body = format_problem(year, season, n, text, choices)
            records.append({
                "year": year,
                "contest": season,
                "number": n,
                "answer": correct,
                "text": body,
            })
        print(f"{slug}: kept last {len(kept)}/{len(qs)} problems")
        time.sleep(0.4)

    with open(out / "problems.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sep = "\n" + "<" + "END" + ">" + "\n"
    corpus = sep.join(r["text"] for r in records) + sep
    (out / "corpus.txt").write_text(corpus)

    print(f"\nwrote {len(records)} problems, {len(corpus):,} chars to {out}/")


if __name__ == "__main__":
    main()
