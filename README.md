# AMC-10 GPT — a from-scratch math-problem generator

A character-level GPT trained **from scratch** (PyTorch only — no pretrained weights, no
`transformers` library) to generate AMC 10–style problems in LaTeX. Runs on Apple silicon
via MPS.

## Dataset

Scraped from [LIVE by Po-Shen Loh](https://live.poshenloh.com/past-contests) (MAA-licensed
problems; LaTeX source extracted from the site's embedded Next.js data).

Slicing rules implemented exactly as specified (per contest, keeping the **last** problems):

| Contests        | Problems kept |
|-----------------|---------------|
| before 2010     | last 5  (#21–25) |
| 2010 – 2019     | last 10 (#16–25) |
| 2020 – 2025     | last 15 (#11–25) |

Corpus: **500 problems** (2000–2025, AMC 10A/10B, incl. 2021 Fall C/D as separate contests),
**199,252 chars**, 97-char vocabulary, split 95/5 train/val. Each document:

```
YEAR: 2021
CONTEST: FallD
PROBLEM: 25

A rectangle with side lengths \(1\) and \(3,\) ... What is \(m+n?\)

(A) 14 (B) 23 (C) 46 (D) 59 (E) 67
```

## Model

From-scratch decoder-only transformer (`model.py`): token + learned positional embeddings,
pre-LayerNorm blocks, fused causal attention (`F.scaled_dot_product_attention`), GELU MLP,
weight-tied head. Config used: 4 layers × 4 heads × 256 dims, context 512, ~3.1M params,
dropout 0.1 (residual/embedding only; MPS SDPA does not support attention dropout).

## Training

`train.py` — AdamW (β = 0.9/0.95, wd 0.1), cosine LR 3e-4 → 3e-5 with 150-step warmup,
grad clip 1.0, batch 32 × 512 tokens, 3000 iters (~49M tokens ≈ 260 epochs over the corpus).

Result: **best val loss 1.067 (perplexity ≈ 2.9)** at iter 1750; checkpoints auto-resume
via `checkpoints/last.pt` (run the same command to continue a 10-min-capped run).

```bash
uv run python train.py --max-iters 3000 --batch-size 32
```

## Sampling

```bash
# "next year's" problem 25
uv run python sample.py --year 2026 --contest A --number 25 --temperature 0.7 --top-k 30

# continue your own seed text (see the seed-continuation snippet in sample.py's docstring)
```

Prompt with any `YEAR/CONTEST/PROBLEM` header (or arbitrary seed text); generation stops at
the `<END>` document boundary.

## Honest limitations

At 3M params on ~200k chars, the model reliably learns **format, structure, and style** —
headers, LaTeX fragments, five choice lines, answer-slot conventions — but generated prose
is not fully coherent, and problems are not guaranteed to be well-posed or solvable.
Scaling data (all 25 problems per contest, AMC 12, more years), context, and parameters
would each improve quality.

## Files

- `scraper.py` — dataset scraper (poshenloh.com → `data/problems.jsonl` + `data/corpus.txt`)
- `model.py` — GPT implementation from scratch
- `train.py` — training loop (MPS/CUDA/CPU, resume, best-checkpoint tracking)
- `sample.py` — generation CLI
- `checkpoints/best.pt` — trained weights (val 1.067)
