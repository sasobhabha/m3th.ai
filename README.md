# m3th — a from-scratch math-problem generator & quiz CLI

A character-level GPT trained **from scratch** (PyTorch only — no pretrained weights, no
`transformers` library) to generate AMC 10–style problems in LaTeX, then quiz you on them.
Runs on Apple silicon via MPS.

## Install (Homebrew)

```bash
brew install sasobhabha/m3th/m3th
```

This bundles a pretrained checkpoint (~13 MB) — no training needed.

```bash
m3th          # interactive REPL: /new generates a problem, /year /contest /number
              # /temp /topk /tokens /seed control it, /show prints settings
m3th quiz     # generates a problem, you answer A–E, it checks the official key,
              # and scores the session (options: -n count, -t temp, -s subject)
m3th --help   # everything else, incl. --ckpt to point at a different checkpoint
```

## Web app (LaTeX practice UI)

A Flask app in the repo root renders problems with MathJax: click A–E to answer, it grades
against the official key, colors your pick vs. the right one, and tracks a session score.

```bash
# from the repo root (checkpoint auto-detected)
uv sync --extra web
uv run python app.py                 # http://127.0.0.1:8080

uv run python app.py --host 0.0.0.0  # expose on your network
uv run python app.py --port 5001     # different port
uv run python app.py --ckpt /path/to/best.pt
```

Not part of the Homebrew formula — `brew install sasobhabha/m3th/m3th` stays a lean CLI.

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

- `app.py` + `templates/index.html` — Flask web app (MathJax LaTeX, click-to-answer, score)
- `m3th/` — installable CLI package (`m3th` REPL, `m3th quiz`, model, and bundled data)
- `scraper.py` — dataset scraper (poshenloh.com → `m3th/data/problems.jsonl` + `corpus.txt`)
- `model.py` / `m3th/model.py` — GPT implementation from scratch
- `train.py` — training loop (MPS/CUDA/CPU, resume, best-checkpoint tracking)
- `sample.py` — generation script
- `Formula/m3th.rb` — Homebrew formula (also published as `sasobhabha/m3th` tap)
- checkpoints are served via [GitHub Releases](https://github.com/sasobhabha/m3th.ai/releases)
  (`best.pt`, val 1.067) — not committed, to keep the repo lean
