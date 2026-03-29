import numpy as np
import sentencepiece as spm
import torch


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        return self.sp.DecodeIds(tokens)

    def build_bpb_luts(self, device='cpu'):
        """
        Build lookup tables for tokenizer-agnostic bits-per-byte (bpb) calculation.
        Matches the parameter-golf challenge's exact method.

        Returns:
            base_bytes      : (vocab,) int16 — byte length of token piece, excluding leading space
            has_leading_space: (vocab,) bool  — token piece starts with '▁' (space prefix)
            is_boundary     : (vocab,) bool   — control / unknown / unused / byte tokens
        """
        vocab_size = self.sp.GetPieceSize()
        base_bytes_np        = np.zeros(vocab_size, dtype=np.int16)
        has_leading_space_np = np.zeros(vocab_size, dtype=np.bool_)
        is_boundary_np       = np.ones(vocab_size,  dtype=np.bool_)

        for token_id in range(vocab_size):
            if (self.sp.IsControl(token_id) or
                    self.sp.IsUnknown(token_id) or
                    self.sp.IsUnused(token_id)):
                continue
            is_boundary_np[token_id] = False
            if self.sp.IsByte(token_id):
                base_bytes_np[token_id] = 1
                continue
            piece = self.sp.IdToPiece(token_id)
            if piece.startswith('▁'):
                has_leading_space_np[token_id] = True
                piece = piece[1:]
            base_bytes_np[token_id] = len(piece.encode('utf-8'))

        return (
            torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary_np,       dtype=torch.bool,  device=device),
        )

    def __len__(self):
        return self.sp.GetPieceSize()
