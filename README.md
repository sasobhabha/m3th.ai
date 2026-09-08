# m3th — math practice app (CLI + web), from scratch and fine-tuned

Two ways to practice AMC 10–style problems:

1. **Web app** — a LoRA fine-tuned LLM (Qwen2.5-0.5B) writes coherent problems,
   MathJax renders the LaTeX, you click A–E, the app grades against the real
   official AMC answer key and tracks your session score.
2. **CLI** (`brew install sasobhabha/m3th/m3th`) — a lean char-level GPT trained
   **from scratch** (no pretrained weights) that also quizzes you on terminal.

## Quickstart (web app)

```bash
# from the repo root
uv sync --extra web

# serve on port 8080 — the trained adapter downloads automatically on first run
uv run python3 app.py                       # http://127.0.0.1:8080

# options
uv run python3 app.py --host 0.0.0.0        # expose on your network
uv run python3 app.py --port 5001           # different port
uv run python3 app.py --adapter PATH        # explicit LoRA adapter dir
```

Then open **http://127.0.0.1:8080** — the first visit loads the model and
generates a problem (~5–15 s on the GPU).

Prefer training your own adapter instead of downloading it?

```bash
uv run python3 finetune.py --epochs 3   # ~10 min on Apple silicon, resumable
```

What you get:

- coherent AMC-style problems in LaTeX (statement **and** choices rendered by MathJax)
- optional topic seed ("a square with side length"), year/contest/number, temperature
- click-to-answer: your pick and the correct one are colored, score in the header
- answers are the **official** AMC key for the sampled (year, contest, number) slot

> Honest caveat: the key is real but the problem is invented, so occasionally the
> "correct" answer won't match the invented question. Treat mismatches with suspicion.

## CLI (Homebrew)

```bash
brew install sasobhabha/m3th/m3th
```

This bundles a pretrained char-GPT checkpoint (~13 MB) — no training needed.

```bash
m3th          # interactive REPL: /new generates a problem, /year /contest /number
              # /temp /topk /tokens /seed control it, /show prints settings
m3th quiz     # generates a problem, you answer A–E, it checks the official key,
              # and scores the session (options: -n count, -t temp, -s subject)
m3th --help   # everything else, incl. --ckpt to point at a different checkpoint
```

The formula ships only the CLI — the web app stays in this repo (heavier deps).

## Dataset

Scraped from [LIVE by Po-Shen Loh](https://live.poshenloh.com/past-contests) (MAA-licensed
problems; LaTeX source extracted from the site's embedded Next.js data).

```bash
uv run python3 scraper.py          # 5/10/15 slicing rules  -> 500 problems
uv run python3 scraper.py --all    # all 25 per contest     -> 1300 problems (fine-tune data)
```

Default slicing rules (per contest, keeping the **last** problems):

| Contests        | Problems kept |
|-----------------|---------------|
| before 2010     | last 5  (#21–25) |
| 2010 – 2019     | last 10 (#16–25) |
| 2020 – 2025     | last 15 (#11–25) |

`--all` grabs all 1300 (2000–2025, AMC 10A/10B, incl. 2021 Fall C/D as separate contests)
with official answer keys.

## Models

### 1. LoRA fine-tuned Qwen (web app) — `finetune.py`

Base: **Qwen/Qwen2.5-0.5B-Instruct**, LoRA r=16 α=32 on all attention + MLP projections
(8.8M trainable params), gradient checkpointing, bf16 autocast on MPS, cosine LR 1e-4,
effective batch 8 (1×8 accum). 3 epochs over 1300 problems ≈ 490 steps (~10 min on M5).

Result: final loss ≈ 0.74 (ppl ≈ 2.1) — coherent AMC prose and LaTeX from a 0.5B model.

```bash
uv run python3 finetune.py --epochs 3     # resumes automatically if interrupted
```

### 2. Char-level GPT from scratch (CLI) — `model.py` + `train.py`

Decoder-only transformer: token + learned positional embeddings, pre-LayerNorm blocks,
fused causal attention (`F.scaled_dot_product_attention`), GELU MLP, weight-tied head.
Config: 4 layers × 4 heads × 256 dims, context 512, ~3.1M params.

Training: AdamW (β = 0.9/0.95, wd 0.1), cosine LR 3e-4 → 3e-5 with warmup, grad clip 1.0,
batch 32 × 512 tokens, 3000 iters. **Best val loss 1.067 (ppl ≈ 2.9)**.

```bash
uv run python3 train.py --max-iters 3000 --batch-size 32    # resumable
```

At 3M params on ~200k chars it learns format/structure/style but not coherent prose —
which is exactly why the web app uses the fine-tuned LLM instead.

## Files

- `app.py` + `templates/index.html` — Flask web app (LoRA LLM, MathJax LaTeX,
  click-to-answer grading, session score)
- `finetune.py` — LoRA fine-tuning of Qwen2.5-0.5B on AMC data (MPS, resumable)
- `m3th/` — installable CLI package (`m3th` REPL, `m3th quiz`, char-GPT model, data)
- `scraper.py` — dataset scraper (`--all` for the fine-tune corpus)
- `model.py` / `m3th/model.py` — from-scratch GPT implementation
- `train.py` — char-GPT training loop (MPS/CUDA/CPU, resume, best-checkpoint)
- `sample.py` — char-GPT generation script
- `Formula/m3th.rb` — Homebrew formula (also published as `sasobhabha/m3th` tap)
- checkpoints/LoRA adapters are gitignored; the trained adapter ships via
  [Releases](https://github.com/sasobhabha/m3th.ai/releases/download/v0.2.0/m3th-qwen-lora.zip)
  and `app.py` downloads it automatically when missing
