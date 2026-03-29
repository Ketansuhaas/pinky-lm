import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv      = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj     = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        att   = (q @ k.transpose(-2, -1)) * scale                      # [B, H, T, T]
        mask  = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att   = att.masked_fill(~mask, float('-inf'))
        att   = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)     # [B, T, C]
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PinkyLM(nn.Module):
    def __init__(self, vocab_size, block_size, embed_dim=64, n_heads=4, n_layers=4):
        super().__init__()
        self.block_size = block_size
        self.tok_embed  = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed  = nn.Embedding(block_size, embed_dim)
        self.blocks     = nn.Sequential(*[TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)])
        self.ln         = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)            # [1, T]
        x   = self.tok_embed(x) + self.pos_embed(pos)                  # [B, T, C]
        return self.head(self.ln(self.blocks(x)))                       # [B, T, vocab]

    @torch.no_grad()
    def generate(self, tokenizer, prompt, steps=200, temperature=1.0, device='cpu'):
        self.eval()
        tokens = tokenizer.encode(prompt)
        for _ in range(steps):
            ctx    = tokens[-self.block_size:]
            x      = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = self(x)[0, -1] / temperature
            token  = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            tokens.append(token)
        return tokenizer.decode(tokens)
