"""Fine-tune Qwen2.5-0.5B-Instruct on AMC 10 problems with LoRA (Apple MPS).

Unlike the from-scratch char-level GPT (model.py), this starts from a real
pretrained LLM so generated problems have coherent English prose — only the
AMC style is learned.

Data: every scraped problem becomes a chat sample:
  system    fixed instructions
  user      generate an AMC 10 problem: year/contest/number (+ difficulty hint)
  assistant the full problem statement + (A)-(E) choices in LaTeX

Run:
    uv run python3 finetune.py --epochs 3
Resume after an interruption (same command; state is checkpointed):
    uv run python3 finetune.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("m3th/data/problems.jsonl")
ADAPTER_DIR = Path("checkpoints/qwen-lora")
STATE = Path("checkpoints/qwen-lora/state.pt")

SYSTEM = (
    "You write original AMC 10-style competition math problems. "
    "Output exactly one problem: a self-contained statement using LaTeX "
    "for all math (inline \\( ... \\)), followed by five answer choices "
    "(A) through (E) on one line. Do not solve the problem."
)

MAX_LEN = 512  # prompt + completion tokens


def build_samples() -> list[dict]:
    records = [json.loads(l) for l in open(DATA)]
    samples = []
    for r in records:
        body = r["text"]
        # strip the training header used by the char model
        lines = body.split("\n", 3)
        problem = lines[3] if len(lines) > 3 else body
        difficulty = "hard" if r["number"] >= 20 else ("medium" if r["number"] >= 11 else "easy")
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content":
                    f"Write an AMC 10 {difficulty} problem, "
                    f"{r['year']} {r['contest']} contest, problem {r['number']}."},
                {"role": "assistant", "content": problem},
            ]
        })
    random.Random(42).shuffle(samples)
    return samples


def encode_batch(samples: list[dict], tok) -> dict[str, torch.Tensor]:
    """Tokenize with the chat template; mask loss on prompt tokens (-100)."""
    input_ids, labels = [], []
    for s in samples:
        prompt = tok.apply_chat_template(s["messages"][:-1], tokenize=False,
                                         add_generation_prompt=True)
        full = tok.apply_chat_template(s["messages"], tokenize=False,
                                       add_generation_prompt=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:MAX_LEN]
        lab = list(f_ids)
        for i in range(min(len(p_ids), len(lab))):
            lab[i] = -100
        input_ids.append(f_ids)
        labels.append(lab)

    pad = tok.pad_token_id or tok.eos_token_id
    maxlen = max(len(x) for x in input_ids)
    ids = torch.full((len(input_ids), maxlen), pad, dtype=torch.long)
    att = torch.zeros((len(input_ids), maxlen), dtype=torch.long)
    lab = torch.full((len(input_ids), maxlen), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(input_ids, labels)):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
        lab[i, :len(y)] = torch.tensor(y)
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-interval", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)  # fp32
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()          # required for LoRA + checkpointing
    model.gradient_checkpointing_enable()       # huge activation-memory win on MPS
    model.to(device)
    model.print_trainable_parameters()

    samples = build_samples()
    print(f"{len(samples)} training samples")

    eff = args.batch_size * args.accum
    steps_per_epoch = math.ceil(len(samples) / eff)
    total_steps = steps_per_epoch * args.epochs
    print(f"batch {args.batch_size} x accum {args.accum} = {eff} "
          f"-> {steps_per_epoch} steps/epoch, {total_steps} total")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    warmup = 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / (total_steps - warmup)))))

    start = 0
    best = float("inf")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        st = torch.load(STATE, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"] + 1
        best = st["best"]
        print(f"resumed at step {start}/{total_steps}")

    rng = random.Random(1234)
    order = list(range(len(samples)))

    model.train()
    for step in range(start, total_steps):
        epoch, offset = divmod(step, steps_per_epoch)
        if offset == 0 and step > 0 or (step == 0 and start == 0):
            rng.shuffle(order)
        batch_idx = order[(offset * eff) % len(samples): (offset * eff) % len(samples) + eff]
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(0, len(batch_idx), args.batch_size):
            chunk = [samples[i] for i in batch_idx[micro:micro + args.batch_size]]
            batch = encode_batch(chunk, tok)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="mps", dtype=torch.bfloat16,
                                enabled=device.type == "mps"):
                out = model(**batch)
                loss = out.loss / max(1, math.ceil(len(batch_idx) / args.batch_size))
            loss.backward()
            loss_sum += loss.item() * max(1, math.ceil(len(batch_idx) / args.batch_size))
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        sched.step()
        if device.type == "mps":
            torch.mps.empty_cache()  # MPS caches aggressively; keeps long runs alive

        if (step + 1) % args.eval_interval == 0 or step == total_steps - 1:
            avg = loss_sum / max(1, len(batch_idx))
            print(f"step {step + 1}/{total_steps} (epoch {epoch + 1}) "
                  f"loss {avg:.4f} ppl {math.exp(avg):.1f} lr {sched.get_last_lr()[0]:.2e}")
            model.save_pretrained(ADAPTER_DIR)
            torch.save({"step": step, "best": best, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "model": model.state_dict()}, STATE)

    model.save_pretrained(ADAPTER_DIR)
    STATE.unlink(missing_ok=True)  # finished
    print(f"done — LoRA adapter in {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
