# pinky-lm

A small transformer language model trained on FineWeb, built for the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) challenge (best bits-per-byte on FineWeb val, model ≤ 16 MB).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your wandb key:
```
WANDB_API_KEY=your_key_here
```

## Data

Download FineWeb data (pre-tokenized with 1024-vocab SentencePiece):
```bash
python3 scripts/download_data.py
```

Expected layout:
```
data/
  tokenizers/
    fineweb_1024_bpe.model
  datasets/
    fineweb10B_sp1024/
      fineweb_train_000000.bin
      fineweb_train_000001.bin
      ...
      fineweb_val_000000.bin
```

## Training

```bash
./scripts/train.sh
```

Or directly:
```bash
python3 train.py \
    --steps 5000 --eval-every 500 \
    --block-size 128 --batch-size 32 --lr 1e-3 \
    --embed-dim 64 --n-heads 4 --n-layers 4 \
    --train data/datasets/fineweb10B_sp1024 \
    --val   data/datasets/fineweb10B_sp1024 \
    --wandb-project pinky-lm --wandb-run-name run-01
```

Checkpoints are saved to `checkpoints/{timestamp}_{run}/`. Best checkpoint (lowest val bpb) is saved as `best.pt`.

## Evaluation

Full val eval on fp32 checkpoint:
```bash
python3 eval.py --checkpoint checkpoints/<run>/best.pt
```

## Quantization

Compress fp32 → int8 + zlib:
```bash
./scripts/quantize.sh checkpoints/<run>/best.pt
# saves best_int8.zlib in same directory
```

Full val eval on quantized model:
```bash
./scripts/eval_quantized.sh checkpoints/<run>/best_int8.zlib
```

## Chat

Run a terminal chatbot with a trained checkpoint:
```bash
python3 chat.py --checkpoint checkpoints/<run>/best.pt
python3 chat.py --checkpoint checkpoints/<run>/best_int8.zlib --quantized
```

Commands during chat: `/temp <float>`, `/steps <int>`, `/quit`

## Model

`PinkyLM` is a standard decoder-only transformer:

| Param       | Default |
|-------------|---------|
| vocab_size  | 1024    |
| block_size  | 128     |
| embed_dim   | 64      |
| n_heads     | 4       |
| n_layers    | 4       |
| params      | ~340K   |

## Metric

Bits per byte (bpb): `(val_loss / log(2)) * (tokens / bytes)`, computed with exact byte counting via SentencePiece lookup tables — matches the parameter-golf challenge evaluation exactly.
