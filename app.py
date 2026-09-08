"""m3th web — a Flask practice app for generated AMC 10-style problems.

Renders problems with MathJax (LaTeX), grades answers against the official
AMC answer key for the sampled (year, contest, number) slot, and tracks a
session score.

Run from the repo root:
    uv run python app.py            # serves on port 8080
    python app.py --port 8080       # any python with flask + the package deps
    python app.py --host 0.0.0.0    # expose on your network
"""

from __future__ import annotations

import argparse
import random
import re
import threading
import uuid

from flask import Flask, redirect, render_template, request, session, url_for

from m3th.__main__ import HEADER_RE, Generator, answers, default_ckpt

# The model output always lays its choices out as "(A) ... (B) ... (E) ...".
CHOICES_RE = re.compile(r"\(A\).*\(E\)", re.S)
LETTERS = "ABCDE"

# Server-side store for the problem currently on screen, keyed by a per-session
# token (problem text is too bulky for a signed cookie).
_PROBLEMS: dict[str, dict] = {}

# torch generate on MPS is not safe under concurrent requests.
_GEN_LOCK = threading.Lock()


def _parse(raw: str) -> dict:
    """Split a raw completion into body + (A)-(E) choice list."""
    m = CHOICES_RE.search(raw)
    if not m:
        return {"has_choices": False, "body": HEADER_RE.sub("", raw).strip(), "choices": []}
    problem_text = raw[: m.start()].rstrip()
    choices_blob = raw[m.start():].rstrip()
    parts = re.split(r"\(([A-E])\)", choices_blob)  # [pre, 'A', txt, 'B', txt, ...]
    choices = list(zip(parts[1::2], (t.strip() for t in parts[2::2])))
    return {"has_choices": True, "body": HEADER_RE.sub("", problem_text).strip(),
            "choices": choices}


def create_app(ckpt: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = uuid.uuid4().hex  # per-process; sessions are ephemeral anyway

    gen = Generator(ckpt or default_ckpt())
    key = answers()
    combos = sorted(key.keys())
    app.extensions["m3th_gen"] = gen

    def combo_from_form() -> tuple[int, str, int]:
        """Slot to sample from, honoring the form; blank year = random slot."""
        year = (request.form.get("year") or "").strip()
        contest = (request.form.get("contest") or "").strip().upper()
        number = (request.form.get("number") or "").strip()
        if not year:
            return random.choice(combos)
        y = int(year)
        c = contest or "A"
        n = int(number) if number else 25
        if (y, c, n) not in key:
            # keep a slot the key actually has, matching the requested year/contest
            same = [k for k in combos if k[0] == y and k[1] == c]
            if not same:
                same = [k for k in combos if k[0] == y]
            (y, c, n) = random.choice(same) if same else random.choice(combos)
        return (y, c, n)

    def score() -> dict:
        return session.setdefault("score", {"asked": 0, "correct": 0})

    def _new_problem(year: int, contest: str, number: int, temp: float, subject: str) -> dict:
        gen.year, gen.contest, gen.number = year, contest, number
        gen.temperature = temp
        with _GEN_LOCK:
            raw = gen.raw(subject) if subject else gen.new_problem()
        token = uuid.uuid4().hex
        session["token"] = token
        p = {"raw": raw, "year": year, "contest": contest, "number": number, "temp": temp,
             "subject": subject, "answered": False, "result": None, **_parse(raw)}
        _PROBLEMS[token] = p
        # keep the store bounded (single-user app, long sessions)
        if len(_PROBLEMS) > 32:
            for old in list(_PROBLEMS)[:-16]:
                _PROBLEMS.pop(old, None)
        return p

    @app.get("/")
    def index():
        token = session.get("token")
        p = _PROBLEMS.get(token) if token else None
        if p is None:  # first visit: generate one now (~10-20 s)
            (y, c, n) = random.choice(combos)
            p = _new_problem(y, c, n, gen.temperature, "")
        settings = session.setdefault(
            "settings", {"year": None, "contest": "", "number": 25, "temp": gen.temperature,
                         "subject": ""})
        return render_template("index.html", p=p, s=settings, score=score())

    @app.post("/next")
    def next_problem():
        form = request.form
        y, c, n = combo_from_form()
        try:
            temp = float(form.get("temp") or gen.temperature)
        except ValueError:
            temp = gen.temperature
        temp = max(0.05, min(2.0, temp))
        subject = (form.get("subject") or "").strip()
        session["settings"] = {"year": form.get("year") or None, "contest": form.get("contest") or "",
                               "number": n, "temp": temp, "subject": subject}
        _new_problem(y, c, n, temp, subject)
        return redirect(url_for("index"))

    @app.post("/answer")
    def answer():
        token = session.get("token")
        p = _PROBLEMS.get(token) if token else None
        letter = (request.form.get("letter") or "").strip().upper()
        if p is None or p["answered"] or letter not in LETTERS or not p["has_choices"]:
            return redirect(url_for("index"))
        correct = key[(p["year"], p["contest"], p["number"])]
        was = letter == correct
        p["answered"] = True
        p["result"] = {"yours": letter, "correct": correct, "was_correct": was}
        # reassign: in-place edits of a nested dict don't flag the cookie session
        sc = score()
        session["score"] = {"asked": sc["asked"] + 1, "correct": sc["correct"] + int(was)}
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        session["score"] = {"asked": 0, "correct": 0}
        return redirect(url_for("index"))

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="m3th web app (Flask) — LaTeX practice UI")
    ap.add_argument("--ckpt", default=None, help="path to checkpoint (default: auto)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ckpt = args.ckpt or default_ckpt()
    if not ckpt.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            "train one (python train.py) or download best.pt from the GitHub releases\n"
            "to ~/.cache/m3th/best.pt, or pass --ckpt /path/to/best.pt"
        )

    app = create_app(str(ckpt))
    print(f"loaded {ckpt} — serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
