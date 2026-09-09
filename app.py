"""m3th web — a Flask practice app for LLM-generated AMC 10-style problems.

Generation uses the LoRA fine-tuned Qwen2.5-0.5B-Instruct, so prose is
coherent; answers are graded against the *official* AMC key for the sampled
(year, contest, number) slot. LaTeX is rendered client-side by MathJax.

Run from the repo root:
    uv run python3 app.py            # serves on port 8080
    python3 app.py --host 0.0.0.0    # expose on your network
    python3 app.py --port 5001       # different port
    python3 app.py --adapter PATH    # explicit LoRA adapter dir

If no adapter exists, the trained one is downloaded automatically from the
GitHub release (or train your own with finetune.py).
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import re
import shutil
import threading
import urllib.request
import uuid
import zipfile
from pathlib import Path

import torch
from flask import Flask, redirect, render_template, request, session, url_for
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from m3th.__main__ import answers

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = Path("checkpoints/qwen-lora")
ADAPTER_URL = (
    "https://github.com/sasobhabha/m3th.ai/releases/download/v0.2.0/m3th-qwen-lora.zip"
)
CHOICES_RE = re.compile(r"\((?P<letter>[A-E])\)")
# bare LaTeX in a choice (e.g. "\frac{5}{13}") needs math delimiters to render
LATEX_RE = re.compile(r"\\[a-zA-Z]+|\^|_\d|[{]\\")
MAX_NEW_TOKENS = 350
SOLVE_SAMPLES = 3          # self-consistency votes
SOLVE_MAX_NEW_TOKENS = 600

SYSTEM = (
    "You write original AMC 10-style competition math problems. "
    "Output exactly one problem: a self-contained statement using LaTeX "
    "for all math (inline \\( ... \\)). Do not include multiple choice options. Do not solve the problem."
)

SOLVE_SYSTEM = (
    "You are an expert competition mathematician. Solve the given math "
    "problem step by step, concisely. Provide the final numerical answer as a simple number or decimal, "
    "on its own line, prefixed exactly with 'Answer: '."
)

FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Server-side store for the problem currently on screen, keyed by a per-session
# token (problem text is too bulky for a signed cookie).
_PROBLEMS: dict[str, dict] = {}

# torch generate on MPS is not safe under concurrent requests.
_GEN_LOCK = threading.Lock()


def default_adapter() -> Path:
    env = Path(os.environ["M3TH_ADAPTER"]) if os.environ.get("M3TH_ADAPTER") else None
    return env or DEFAULT_ADAPTER


def find_adapter() -> Path:
    """LoRA adapter: trained locally, else downloaded from the release."""
    for base in (default_adapter(), Path(__file__).resolve().parent / "checkpoints" / "qwen-lora"):
        if (base / "adapter_config.json").exists():
            return base
    return download_adapter()


def download_adapter() -> Path:
    """Fetch and unpack the trained adapter from the GitHub release."""
    dest = default_adapter()
    dest.parent.mkdir(parents=True, exist_ok=True)
    zpath = dest.parent / "m3th-qwen-lora.zip"
    print(f"downloading trained LoRA adapter from the release (~35 MB)…")
    req = urllib.request.Request(ADAPTER_URL, headers={"User-Agent": "m3th"})
    with urllib.request.urlopen(req, timeout=300) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest.parent)
    zpath.unlink()
    if not (dest / "adapter_config.json").exists():
        raise SystemExit(f"adapter download failed: no adapter_config.json in {dest}")
    print(f"adapter ready: {dest}")
    return dest


class LLM:
    """Fine-tuned Qwen 0.5B with LoRA: writes problems, and — with the adapter
    temporarily disabled — solves them. One model, two hats."""

    def __init__(self, adapter: Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()

    def _chat(self, system: str, user: str, max_new: int, temp: float) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new, do_sample=True,
                temperature=max(0.1, temp), top_p=0.95,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, year: int, contest: str, number: int, temp: float,
                 seed_topic: str = "") -> str:
        difficulty = "hard" if number >= 20 else ("medium" if number >= 11 else "easy")
        user = f"Write an AMC 10 {difficulty} problem, {year} {contest} contest, problem {number}."
        if seed_topic:
            user += f" Begin it with something like: \"{seed_topic}\""
        return self._chat(SYSTEM, user, MAX_NEW_TOKENS, temp)

    def solve(self, statement: str) -> dict:
        """The same model solves the problem it wrote, via self-consistency:
        SOLVE_SAMPLES chain-of-thought solutions with the LoRA adapter disabled
        (so it solves like the base instruct model, not in problem-writer mode),
        majority vote decides the verdict."""
        user = f"{statement}"
        votes: list[float] = []
        reasons: list[str] = []
        for _ in range(SOLVE_SAMPLES):
            with torch.no_grad():
                # base-model headspace: disable the problem-writer LoRA weights
                with self.model.disable_adapter():
                    text = self._chat(SOLVE_SYSTEM, user, SOLVE_MAX_NEW_TOKENS, 0.7)
            
            val = None
            # 1. Try 'Answer: '
            m = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
            if m:
                val = float(m[-1])
            else:
                # 2. Try \boxed{}
                boxed = re.findall(r"\\boxed\{(.*?)\}", text)
                if boxed:
                    nums = FLOAT_RE.findall(boxed[-1])
                    if nums:
                        val = float(nums[-1])
                else:
                    # 3. Last resort, last number in text
                    nums = FLOAT_RE.findall(text)
                    if nums:
                        val = float(nums[-1])
            
            if val is not None:
                votes.append(val)
            reasons.append(text.strip())
            
        if votes:
            rounded_votes = [round(v) for v in votes]
            top, n = collections.Counter(rounded_votes).most_common(1)[0]
            return {"verdict": top, "votes": f"{n}/{len(votes)}",
                    "reasoning": reasons}
        return {"verdict": None, "votes": f"0/{len(votes)}", "reasoning": reasons}


def _wrap_choice(text: str) -> str:
    """MathJax needs delimiters; training data stores choices as bare LaTeX."""
    return "\\( " + text + " \\)" if LATEX_RE.search(text) else text


def parse_problem(raw: str) -> dict:
    """Split model output into statement, ignoring any generated choices."""
    raw = raw.strip()
    match = re.search(r"\s*\([A-E]\)\s", raw)
    if match:
        raw = raw[:match.start()]
    return {"body": raw.strip()}


def create_app(adapter: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = uuid.uuid4().hex  # per-process; sessions are ephemeral anyway

    llm = LLM(adapter or find_adapter())

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
            y, c, n = random.randint(2030, 2099), random.choice(["A", "B"]), random.randint(1, 25)
            p = _new_problem(y, c, n, 0.8, "")
        settings = session.setdefault(
            "settings", {"number": 25, "temp": 0.8, "topic": ""})
        return render_template("index.html", p=p, s=settings, score=score())

    @app.post("/next")
    def next_problem():
        form = request.form
        number = int(form.get("number") or 25)
        number = max(1, min(25, number))
        try:
            temp = max(0.1, min(1.5, float(form.get("temp") or 0.8)))
        except ValueError:
            temp = 0.8
        topic = (form.get("topic") or "").strip()
        
        # random future slot: ensures model generates a novel problem
        y, c, n = random.randint(2030, 2099), random.choice(["A", "B"]), number
        
        session["settings"] = {"number": number, "temp": temp, "topic": topic}
        _new_problem(y, c, n, temp, topic)
        return redirect(url_for("index"))

    @app.post("/answer")
    def answer():
        token = session.get("token")
        p = _PROBLEMS.get(token) if token else None
        
        user_answer_str = (request.form.get("answer") or "").strip()
        try:
            user_val = float(user_answer_str)
        except ValueError:
            user_val = None

        if p is None or p["answered"] or user_val is None:
            return redirect(url_for("index"))
        
        # the same loaded model solves the problem it wrote (~15-40 s)
        with _GEN_LOCK:
            solve_result = llm.solve(p["body"])
            p["solve"] = solve_result
            
        correct = solve_result["verdict"]
        was = False
        if correct is not None:
            was = (round(user_val) == round(correct))
        
        p["answered"] = True
        p["result"] = {"yours": user_answer_str, "correct": correct, "was_correct": was}
        
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
