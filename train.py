"""Train the from-scratch char-level GPT on the AMC 10 corpus (Apple MPS)."""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT

END = "<END>"  # document separator used by the scraper


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, chars


def encode(s: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids, itos: list[str]) -> str:
    return "".join(itos[i] for i in ids)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=40):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--no-resume", action="store_true", help="ignore existing last.pt and start over")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    text = Path("data/corpus.txt").read_text()
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.95 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(data):,} tokens, vocab {len(itos)}, train {n:,} / val {len(data)-n:,}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    model = GPT(
        vocab_size=len(itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    print(f"model params (non-emb): {model.num_params():,}")

    # cosine schedule with linear warmup
    def lr_at(it: int) -> float:
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        t = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return 0.1 * args.lr + 0.5 * (1.0 - 0.1) * args.lr * (1 + math.cos(math.pi * t))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    start_iter = 0
    best_val = float("inf")
    last_path = out_dir / "last.pt"
    if last_path.exists() and not args.no_resume:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_iter = ckpt["iter"]
        best_val = ckpt["best_val"]
        print(f"resumed from {last_path} at iter {start_iter} (best val {best_val:.4f})")

    t0 = time.time()
    for it in range(start_iter + 1, args.max_iters + 1):
        closed_form_lr = lr_at(it)
        for g in opt.param_groups:
            g["lr"] = closed_form_lr
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 50 == 0:
            speed = it / (time.time() - t0)
            print(f"iter {it:5d}/{args.max_iters}  train loss {loss.item():.4f}  lr {opt.param_groups[0]['lr']:.2e}  ({speed:.1f} it/s)")

        if it % args.eval_interval == 0 or it == args.max_iters:
            vl = estimate_loss(model, val_data, args.block_size, args.batch_size, device, iters=args.eval_iters)
            print(f"== iter {it}: val loss {vl:.4f} (perplexity {math.exp(vl):.2f})")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "iter": it,
                    "best_val": best_val,
                },
                last_path,
            )
            if vl < best_val:
                best_val = vl
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": {
                            "vocab_size": len(itos),
                            "block_size": args.block_size,
                            "n_layer": args.n_layer,
                            "n_head": args.n_head,
                            "n_embd": args.n_embd,
                            "dropout": args.dropout,
                        },
                        "itos": itos,
                        "val_loss": vl,
                        "iter": it,
                    },
                    out_dir / "best.pt",
                )
                print("   saved checkpoint (best)")

    print(f"done in {time.time()-t0:.0f}s, best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
