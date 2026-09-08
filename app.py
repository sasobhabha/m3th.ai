"""m3th web — a Flask practice app for LLM-generated AMC 10-style problems.

Generation uses the LoRA fine-tuned Qwen2.5-0.5B-Instruct (finetune.py), so
prose is coherent; answers are graded against the *official* AMC key for the
sampled (year, contest, number) slot. LaTeX is rendered client-side by MathJax.

Run from the repo root:
    uv run python3 app.py            # serves on port 8080
    python3 app.py --host 0.0.0.0    # expose on your network
    python3 app.py --port 5001       # different port
    python3 app.py --ckpt PATH       # explicit LoRA adapter dir
"""

from __future__ import annotations

import argparse
import os
import random
import re
import threading
import uuid
from pathlib import Path

import torch
from flask import Flask, redirect, render_template, request, session, url_for
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3th.__main__ import answers

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = Path("checkpoints/qwen-lora")
CHOICES_RE = re.compile(r"\((?P<letter>[A-E])\)")
# bare LaTeX in a choice (e.g. "\frac{5}{13}") needs math delimiters to render
LATEX_RE = re.compile(r"\\[a-zA-Z]+|\^|_\d|[{]\\")
LETTERS = "ABCDE"
MAX_NEW_TOKENS = 350

SYSTEM = (
    "You write original AMC 10-style competition math problems. "
    "Output exactly one problem: a self-contained statement using LaTeX "
    "for all math (inline \\( ... \\)), followed by five answer choices "
    "(A) through (E) on one line. Do not solve the problem."
)

# Server-side store for the problem currently on screen, keyed by a per-session
# token (problem text is too bulky for a signed cookie).
_PROBLEMS: dict[str, dict] = {}

# torch generate on MPS is not safe under concurrent requests.
_GEN_LOCK = threading.Lock()


def default_adapter() -> Path:
    env = Path(os.environ["M3TH_ADAPTER"]) if os.environ.get("M3TH_ADAPTER") else None
    return env or DEFAULT_ADAPTER


def find_adapter() -> Path:
    """LoRA adapter trained by finetune.py."""
    for base in (default_adapter(), Path(__file__).resolve().parent / "checkpoints" / "qwen-lora"):
        if (base / "adapter_config.json").exists():
            return base
    raise SystemExit(
        f"LoRA adapter not found (looked in {default_adapter()}).\n"
        "Train it with:  uv run python3 finetune.py --epochs 3"
    )


class LLM:
    """Fine-tuned Qwen 0.5B with LoRA, sampling one AMC-style problem."""

    def __init__(self, adapter: Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()

    def generate(self, year: int, contest: str, number: int, temp: float,
                 seed_topic: str = "") -> str:
        difficulty = "hard" if number >= 20 else ("medium" if number >= 11 else "easy")
        user = f"Write an AMC 10 {difficulty} problem, {year} {contest} contest, problem {number}."
        if seed_topic:
            user += f" Begin it with something like: \"{seed_topic}\""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=max(0.1, temp), top_p=0.95,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def _wrap_choice(text: str) -> str:
    """MathJax needs delimiters; training data stores choices as bare LaTeX."""
    return "\\( " + text + " \\)" if LATEX_RE.search(text) else text


def parse_problem(raw: str) -> dict:
    """Split model output into statement + (A)-(E) choice list."""
    raw = raw.strip()
    # choices start at the first standalone (A) that has five letters total
    positions: dict[str, int] = {}
    for m in CHOICES_RE.finditer(raw):
        positions.setdefault(m.group("letter"), m.start())
    if len(positions) == 5 and positions["A"] > 0:
        statement = raw[: positions["A"]].rstrip()
        blob = raw[positions["A"]:]
        parts = re.split(r"\(([A-E])\)", blob)  # ['', 'A', txt, 'B', txt, ...]
        pairs = [(L, _wrap_choice(t.strip())) for L, t in zip(parts[1::2], parts[2::2])]
        if len(pairs) == 5 and all(t for _, t in pairs):
            return {"has_choices": True, "body": statement, "choices": pairs}
    return {"has_choices": False, "body": raw, "choices": []}


def create_app(adapter: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = uuid.uuid4().hex  # per-process; sessions are ephemeral anyway

    llm = LLM(adapter or find_adapter())
    key = answers()
    combos = sorted(key.keys())
    print(f"answer key: {len(combos)} slots")

    def score() -> dict:
        return session.setdefault("score", {"asked": 0, "correct": 0})

    def _new_problem(year: int, contest: str, number: int, temp: float, topic: str) -> dict:
        with _GEN_LOCK:  # one MPS generation at a time (~5-15 s)
            raw = llm.generate(year, contest, number, temp, topic)
        token = uuid.uuid4().hex
        session["token"] = token
        p = {"year": year, "contest": contest, "number": number, "temp": temp,
             "topic": topic, "answered": False, "result": None, **parse_problem(raw)}
        _PROBLEMS[token] = p
        if len(_PROBLEMS) > 32:  # single-user app: keep the store bounded
            for old in list(_PROBLEMS)[:-16]:
                _PROBLEMS.pop(old, None)
        return p

    @app.get("/")
    def index():
        token = session.get("token")
        p = _PROBLEMS.get(token) if token else None
        if p is None:  # first visit: generate one now (~5-15 s)
            (y, c, n) = random.choice(combos)
            p = _new_problem(y, c, n, 0.8, "")
        settings = session.setdefault(
            "settings", {"year": None, "contest": "", "number": 25, "temp": 0.8, "topic": ""})
        return render_template("index.html", p=p, s=settings, score=score())

    @app.post("/next")
    def next_problem():
        form = request.form
        year = (form.get("year") or "").strip()
        contest = (form.get("contest") or "").strip().upper() or "A"
        number = int(form.get("number") or 25)
        number = max(1, min(25, number))
        try:
            temp = max(0.1, min(1.5, float(form.get("temp") or 0.8)))
        except ValueError:
            temp = 0.8
        topic = (form.get("topic") or "").strip()
        if year:  # honor the requested year/contest/number as written
            y, c, n = int(year), contest, number
        else:  # random official-key slot: keeps every problem gradeable
            y, c, n = random.choice(combos)
        session["settings"] = {"year": year or None, "contest": contest,
                               "number": number, "temp": temp, "topic": topic}
        _new_problem(y, c, n, temp, topic)
        return redirect(url_for("index"))

    @app.post("/answer")
    def answer():
        token = session.get("token")
        p = _PROBLEMS.get(token) if token else None
        letter = (request.form.get("letter") or "").strip().upper()
        if p is None or p["answered"] or letter not in LETTERS or not p["has_choices"]:
            return redirect(url_for("index"))
        correct = key.get((p["year"], p["contest"], p["number"]))
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
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: checkpoints/qwen-lora)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    app = create_app(Path(args.adapter) if args.adapter else None)
    print(f"serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
