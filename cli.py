"""Interactive CLI for generating AMC 10-style problems from a trained checkpoint.

Usage:
    uv run python cli.py [--ckpt checkpoints/best.pt]

Commands (type them at the prompt):
    /new                     generate a problem with current settings
    /year 2026               set the YEAR header used by /new
    /contest A               set the CONTEST header (A, B, FallC, ...)
    /number 25               set the PROBLEM number
    /seed <text>             continue your own seed text instead of a header
    /temp 0.7                sampling temperature (0.1-1.5)
    /topk 30                 top-k sampling (0 = off)
    /tokens 1200             max tokens to generate
    /show                    show current settings
    /quit                    exit
"""

import argparse

import torch

from model import GPT

END = "<END>"


class Generator:
    def __init__(self, ckpt_path: str):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
