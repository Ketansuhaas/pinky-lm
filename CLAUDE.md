# pinky-lm — Claude Code Guide

## What this project is

Competitive entry for the **OpenAI Parameter Golf** challenge: train the best language model that fits in **≤16 MB** (quantized), evaluated by **bits-per-byte (bpb)** on the FineWeb validation set. Lower bpb = better. This is a pure compression/modeling competition — no chat, no RLHF, just next-token prediction quality per byte.

The challenge submission is `best_int8.zlib` — the fp32 model quantized to int8 and zlib-compressed.

## Primary metric

**bpb = (val_loss / log(2)) × (tokens / bytes)**

Measured with exact SentencePiece byte counting via 3 lookup tables (`base_bytes`, `has_leading_space`, `is_boundary`). These are in `src/tokenizer.py:build_bpb_luts()` and must exactly match the challenge's evaluation. Never change this formula.

A rough baseline for reference: a random model gives ~10 bpb. A decent small model should get below 2.5 bpb. Parameter-golf winners typically get under 2.0.

## Architecture

`src/model.py` — `PinkyLM`: standard decoder-only transformer
- Vocab: 1024 (SentencePiece BPE, FineWeb)
- Block size: 1024 tokens
- Default config: embed_dim=64, n_heads=4, n_layers=4, ~340K params

`src/trainer.py` — training loop with wandb logging, bpb tracking, checkpoint management
`src/dataset.py` — `BinDataset` (memmap uint16 shards) + `ConcatDataset` for multi-shard
`src/tokenizer.py` — SentencePiece wrapper + bpb LUTs
`src/quantization.py` — int8 per-row quantization + zlib compression (ported from challenge)

## Key workflows

```bash
# Train
./scripts/train.sh

# Full val eval (fp32)
python3 eval.py --checkpoint checkpoints/<run>/best.pt

# Quantize fp32 → int8+zlib
./scripts/quantize.sh checkpoints/<run>/best.pt

# Eval quantized model
./scripts/eval_quantized.sh checkpoints/<run>/best_int8.zlib

# Interactive chatbot
python3 chat.py --checkpoint checkpoints/<run>/best.pt
python3 chat.py --checkpoint checkpoints/<run>/best_int8.zlib --quantized
```

## Constraints

- **Submission must be ≤16 MB** after int8+zlib quantization. Always check size after quantizing.
- The quantization scheme in `src/quantization.py` is fixed — it must match the challenge's dequantization code exactly.
- The bpb formula in `src/trainer.py:_evaluate()` and `eval.py` must stay in sync.
- Tokenizer is fixed: `data/tokenizers/fineweb_1024_bpe.model` (1024-vocab SentencePiece BPE).

## Proactive suggestions — always look for these

### When reviewing model architecture (`src/model.py`)
- Is the parameter budget being used efficiently? Check if increasing depth (n_layers) vs width (embed_dim) would help. Depth tends to be more parameter-efficient.
- Are there obvious wins like weight tying (tie input embedding to output head)? This saves vocab_size × embed_dim parameters for free — ~65K params at current settings.
- Is there a better activation than GELU? SwiGLU or ReLU² can outperform GELU at small scale.
- FFN expansion ratio is 4x — at small embed dims, this may be wasteful. Consider 8x with smaller embed_dim.
- Rotary positional embeddings (RoPE) vs learned absolute — RoPE tends to generalize better.
- Flash attention could speed up training significantly at block_size=1024.

### When reviewing training config (`scripts/train.sh`, `src/trainer.py`)
- Is the learning rate appropriate for current model size? Smaller models often need higher lr (3e-3 to 1e-2).
- Is a cosine LR schedule being used? Flat lr wastes steps near the end of training.
- Gradient clipping at 1.0 is standard but may need tuning.
- Consider warmup steps — sudden large gradients at step 0 can hurt.
- With block_size=1024 and batch_size=32, memory usage may be high — check if batch_size needs reducing on the current device.
- More steps almost always helps — suggest increasing if compute allows.

### When reviewing the quantization pipeline
- After quantizing, always verify the quantized bpb is close to fp32 bpb (degradation >0.05 bpb is a red flag).
- Check that `best_int8.zlib` is actually under 16 MB before considering it a valid submission.
- Larger models that still fit in 16 MB after quantization are always better — compute what the max fp32 size is that compresses to ≤16 MB (typically ~50–60 MB fp32 → 16 MB quantized+zlib).

### When looking at checkpoints / training runs
- Compare `val_bpb` across runs — always suggest what changed and whether the trend is improving.
- If val_bpb stopped improving, suggest reducing lr, increasing model size, or more data.
- If train_bpb << val_bpb, the model may be overfitting — unusual for LMs on large datasets but possible with tiny models on few steps.

### When the user asks about improving the model
Always frame suggestions around the challenge constraint:
1. **Biggest bpb gains**: more parameters (up to 16 MB limit), longer training, better architecture
2. **Free wins**: weight tying, better LR schedule, mixed precision
3. **Research-level**: better tokenization, data curriculum, ensemble methods

## Data

`data/datasets/fineweb10B_sp1024/` — pre-tokenized FineWeb shards
- `fineweb_train_000000.bin` ... multiple train shards (uint16, raw token ids)
- `fineweb_val_000000.bin` — single val shard

The dataset is already shuffled. `shuffle=False` in DataLoader is intentional.
`drop_last=True` during training (consistent batch sizes), `drop_last=False` for eval (full coverage).

## What good bpb looks like

| Model | bpb (approx) |
|---|---|
| Random (uniform over 1024 tokens) | ~10.0 |
| Unigram frequency model | ~5.0 |
| Our baseline (340K params) | ~2.8–3.2 |
| Good small model | ~2.3–2.5 |
| Parameter-golf top entries | <2.0 |

When the user runs eval, immediately compare against these ranges and comment on where the model sits.

## Environment

- Device auto-detected: CUDA > MPS > CPU
- `.env` holds `WANDB_API_KEY` — loaded in `train.py` at startup
- Python venv at `.venv/`
- All scripts are run from repo root: `cd /Users/ketan/Desktop/training/self/pinky-lm`
