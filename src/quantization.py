"""
Int8 quantization and dequantization for model checkpoints.
Ported from the parameter-golf challenge's train_gpt.py.

Scheme:
- 2D float tensors (matrices): per-row int8 with fp16 scale
- Other float tensors (vectors/scalars): per-tensor int8
- Small float tensors (<=65536 elements): kept as fp16
- Non-float tensors: passed through unchanged
"""

import io
import zlib

import torch
from torch import Tensor

INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_PERCENTILE        = 99.99984
INT8_CLIP_Q                 = INT8_CLIP_PERCENTILE / 100.0


def _keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix('torch.')
    return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()


def _quantize_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale   = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q       = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale    = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q        = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict(state_dict: dict) -> dict:
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        if not t.is_floating_point():
            passthrough[name] = t
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = _keep_float_tensor(name, t, passthrough_orig_dtypes)
            continue

        q, s = _quantize_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {'scheme': 'per_row', 'axis': 0}
        quantized[name] = q
        scales[name]    = s
        dtypes[name]    = str(t.dtype).removeprefix('torch.')

    return {
        '__quant_format__'        : 'int8_clean_per_row_v1',
        'quantized'               : quantized,
        'scales'                  : scales,
        'dtypes'                  : dtypes,
        'passthrough'             : passthrough,
        'qmeta'                   : qmeta,
        'passthrough_orig_dtypes' : passthrough_orig_dtypes,
    }


def dequantize_state_dict(obj: dict) -> dict:
    out: dict[str, Tensor] = {}
    qmeta                  = obj.get('qmeta', {})
    passthrough_orig_dtypes = obj.get('passthrough_orig_dtypes', {})

    for name, q in obj['quantized'].items():
        dtype = getattr(torch, obj['dtypes'][name])
        s     = obj['scales'][name]
        if qmeta.get(name, {}).get('scheme') == 'per_row' or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()

    for name, t in obj['passthrough'].items():
        out_t     = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


def compress(obj: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return zlib.compress(buf.getvalue(), level=9)


def decompress(data: bytes) -> dict:
    return torch.load(io.BytesIO(zlib.decompress(data)), map_location='cpu')
