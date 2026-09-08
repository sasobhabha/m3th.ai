"""m3th — generate and quiz AMC 10-style problems from a trained char-level GPT.

Usage:
    m3th [--ckpt PATH]
    m3th quiz [--ckpt PATH] [-n COUNT] [-t TEMP] [-s SUBJECT]

A separate Flask web app lives in the repo root: app.py (not part of the brew CLI).

Commands inside the generate REPL: /new /year /contest /number /seed
/temp /topk /tokens /show /help /quit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from importlib import resources
from pathlib import Path

import torch

from .model import GPT

END = "<END>"

# LaTeX-free training header format: model always opens with this block.
HEADER_RE = re.compile(r"^YEAR: (\d+)\nCONTEST: (\w+)\nPROBLEM: (\d+)\n\n")

_ANSWERS = None  # lazy: {(year, contest, number): letter}


def data_dir() -> Path:
    """Packaged dataset, falling back to the repo checkout."""
    try:
        return Path(str(resources.files("m3th") / "data"))
    except Exception:
        return Path(__file__).resolve().parent.parent / "data"


def default_ckpt() -> Path:
    env = os.environ.get("M3TH_CKPT")
    if env:
        return Path(env)
    cache = Path.home() / ".cache" / "m3th" / "best.pt"
    if cache.exists():
        return cache
    # repo checkout / CWD fallback (dev usage)
    for base in (Path(__file__).resolve().parent.parent, Path.cwd()):
        local = base / "checkpoints" / "best.pt"
        if local.exists():
            return local
    return Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"


def answers() -> dict[tuple[int, str, int], str]:
    global _ANSWERS
    if _ANSWERS is None:
        _ANSWERS = {}
        with open(data_dir() / "problems.jsonl") as f:
            for line in f:
                r = json.loads(line)
                _ANSWERS[(r["year"], r["contest"], r["number"])] = r["answer"]
    return _ANSWERS


class Generator:
    def __init__(self, ckpt_path: str | Path):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ck["config"]
        self.model = GPT(**cfg).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        self.itos = ck["itos"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        # sampling state
        self.year = 2026
        self.contest = "A"
        self.number = 25
        self.temperature = 0.7
        self.topk = 30
        self.max_tokens = 1200

    def generate(self, prompt: str) -> str:
        idx = torch.tensor([[self.stoi[c] for c in prompt]], dtype=torch.long, device=self.device)
        topk = self.topk if self.topk > 0 else None
        out = self.model.generate(idx, self.max_tokens, temperature=self.temperature, top_k=topk)
        return "".join(self.itos[i] for i in out[0].tolist()).split(END)[0]

    def new_problem(self) -> str:
        prompt = f"YEAR: {self.year}\nCONTEST: {self.contest}\nPROBLEM: {self.number}\n\n"
        return self.generate(prompt)

    def raw(self, subject: str | None) -> str:
        """Generate a raw completion starting right after the header."""
        prompt = f"YEAR: {self.year}\nCONTEST: {self.contest}\nPROBLEM: {self.number}\n\n"
        if subject:
            prompt += subject + " "
        return self.generate(prompt)


HELP = """commands:
  /new                generate a problem with current settings
  /year N             set YEAR header (e.g. 2026)
  /contest X          set CONTEST header (A, B, FallC, ...)
  /number N           set PROBLEM header
  /seed <text>        continue your own seed text
  /temp F             temperature (default 0.7)
  /topk N             top-k sampling, 0 disables (default 30)
  /tokens N           max generation length (default 1200)
  /show               show settings
  /help               this help
  /quit               exit"""


def cmd_generate(args) -> int:
    gen = Generator(args.ckpt)
    print(f"loaded {args.ckpt} on {gen.device} — type /help for commands, /quit to exit\n")
    print(gen.new_problem())
    print()

    while True:
        try:
            line = input("amc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/new":
                print(gen.new_problem()); print()
            elif cmd == "/year":
                gen.year = int(rest)
            elif cmd == "/contest":
                gen.contest = rest
            elif cmd == "/number":
                gen.number = int(rest)
            elif cmd == "/seed":
                print(gen.generate(rest + " ")); print()
            elif cmd == "/temp":
                gen.temperature = max(0.05, min(2.0, float(rest)))
            elif cmd == "/topk":
                gen.topk = max(0, int(rest))
            elif cmd == "/tokens":
                gen.max_tokens = max(32, min(4096, int(rest)))
            elif cmd == "/show":
                print(f"year={gen.year} contest={gen.contest} number={gen.number} "
                      f"temp={gen.temperature} topk={gen.topk} tokens={gen.max_tokens}")
            else:
                print(f"unknown command {cmd!r} — /help")
        except ValueError as e:
            print("bad value:", e)
    return 0


def _render(problem: str) -> str:
    """Strip the training header and print the problem body + choices."""
    body = HEADER_RE.sub("", problem).strip()
    print(body)
    print()


def cmd_quiz(args) -> int:
    gen = Generator(args.ckpt)
    if args.temp is not None:
        gen.temperature = args.temp
    key = answers()
    print(f"loaded {args.ckpt} on {gen.device} — {args.n} questions, "
          f"answer with A–E (or q to quit)\n")

    score = 0
    asked = 0
    combos = sorted(key.keys())  # every (year, contest, number) with an official key
    for i in range(args.n):
        gen.year, gen.contest, gen.number = random.choice(combos)
        raw = gen.raw(args.subject)

        m = re.search(r"\(A\).*\(E\)", raw, re.S)
        if not m:
            print(f"[{i + 1}/{args.n}] (model output had no choices — skipped)\n")
            print(raw, "\n")
            continue
        problem = raw[: m.start()].rstrip()
        choices = raw[m.start():].rstrip()

        print(f"[{i + 1}/{args.n}] " + "-" * 50)
        _render(problem)
        print(choices, "\n")

        try:
            resp = input("your answer> ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if resp in ("Q", "QUIT", "EXIT"):
            break
        if resp not in "ABCDE" or len(resp) != 1:
            print("please answer A, B, C, D or E\n")
            continue

        asked += 1
        # The model only invented the question; the answer key is the real
        # official one for this (year, contest, number) slot.
        correct = key[(gen.year, gen.contest, gen.number)]
        if resp == correct:
            score += 1
            print(f"✅ correct! ({score}/{asked})\n")
        else:
            print(f"❌ correct answer was {correct} ({score}/{asked})\n")

    if asked:
        pct = 100.0 * score / asked
        print(f"score: {score}/{asked} ({pct:.0f}%) — random guessing is 20%")
    else:
        print("no answers graded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="m3th", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="path to checkpoint (default: ~/.cache/m3th/best.pt)")
    sub = ap.add_subparsers(dest="command")

    qp = sub.add_parser("quiz", help="generate a problem, you answer, it checks the key")
    qp.add_argument("--ckpt", default=None)
    qp.add_argument("-n", type=int, default=5, help="number of questions (default 5)")
    qp.add_argument("-t", "--temp", type=float, default=None, help="sampling temperature")
    qp.add_argument("-s", "--subject", default=None,
                    help="seed text, e.g. 'A square with side length'")

    args = ap.parse_args(argv)
    ckpt = args.ckpt or default_ckpt()
    args.ckpt = str(ckpt)
    if not Path(ckpt).exists():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        print("train one with `m3th-train` or download best.pt to ~/.cache/m3th/", file=sys.stderr)
        return 1

    if args.command == "quiz":
        return cmd_quiz(args)
    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
