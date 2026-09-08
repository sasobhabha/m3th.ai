"""Generate synthetic AMC 10-style problems from a trained checkpoint."""

import argparse

import torch

from m3th.model import GPT

END = "<END>"


def load_model(path: str, device: torch.device) -> tuple[GPT, list[str], dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = GPT(**cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck["itos"], ck


def make_prompt(year: int, contest: str, number: int) -> str:
    return f"YEAR: {year}\nCONTEST: {contest}\nPROBLEM: {number}\n\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--contest", default="A")
    p.add_argument("--number", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=1200)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, itos, ck = load_model(args.ckpt, device)
    stoi = {c: i for i, c in enumerate(itos)}
    print(f"loaded {args.ckpt} (iter {ck['iter']}, val {ck['val_loss']:.4f})")

    prompt = make_prompt(args.year, args.contest, args.number)
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    text = "".join(itos[i] for i in out[0].tolist())
    text = text.split(END)[0]  # stop at document boundary
    print(text)


if __name__ == "__main__":
    main()
