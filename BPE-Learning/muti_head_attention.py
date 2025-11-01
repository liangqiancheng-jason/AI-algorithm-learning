"""
Tiny Transformer attention playground.

The goal of this file is to expose the moving parts of multi-head attention in
PyTorch: the scaled dot-product core, the head-splitting helpers, and several
masking patterns. Detailed comments walk through tensor shapes step by step so
you can follow the data flow while experimenting.
"""

import math
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_header(log_stream: TextIO, title: str) -> None:
    print(f"\n=== {title} ===", file=log_stream)


def _log_tensor(log_stream: TextIO, name: str, tensor: torch.Tensor) -> None:
    print(f"{name} (shape={tuple(tensor.shape)}):", file=log_stream)
    print(tensor, file=log_stream)


def _log_mask(log_stream: TextIO, name: str, mask: torch.Tensor) -> None:
    print(f"{name} (shape={tuple(mask.shape)}, dtype={mask.dtype}):", file=log_stream)
    print(mask, file=log_stream)

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    - Supports an additive mask where masked positions are set to -inf before softmax.
    """
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q,
        K,
        V,
        attn_mask=None,
        *,
        verbose: bool = False,
        log_stream: Optional[TextIO] = None,
    ):
        # Q, K, V: (batch_size, num_heads, sequence_length, head_dim)
        head_dim = Q.size(-1)

        if verbose and log_stream is not None:
            _log_header(log_stream, "Scaled Dot-Product Attention")
            _log_tensor(log_stream, "Q", Q)
            _log_tensor(log_stream, "K", K)
            _log_tensor(log_stream, "V", V)

        # Pairwise dot products between each query and key vector.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)  # (batch_size, num_heads, query_len, key_len)

        if attn_mask is not None:
            # attn_mask shape should be broadcastable to scores: (batch_size, 1, query_len, key_len) or (1, 1, query_len, key_len)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        if verbose and log_stream is not None:
            _log_tensor(log_stream, "Scores before softmax", scores)

        # Distributions over keys for every query position.
        attn = F.softmax(scores, dim=-1)            # (batch_size, num_heads, query_len, key_len)
        attn = self.dropout(attn)
        # Weighted average of value vectors.
        context = torch.matmul(attn, V)             # (batch_size, num_heads, query_len, head_dim)

        if verbose and log_stream is not None:
            _log_tensor(log_stream, "Attention weights", attn)
            _log_tensor(log_stream, "Context", context)

        return context, attn                        # return both for inspection


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    Args:
        d_model: model hidden size
        num_heads: number of heads (d_model must be divisible by num_heads)
        dropout: dropout on attention weights and output projection
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # Reshape the last dimension so each head gets its chunk of size d_k.
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # Inverse of _split_heads: concatenate per-head results back together.
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    @staticmethod
    def make_causal_mask(seq_len, device=None):
        # True where we want to MASK (i.e., disallow attending to future tokens).
        i = torch.arange(seq_len, device=device)[:, None]
        j = torch.arange(seq_len, device=device)[None, :]
        return j > i  # (seq_len, seq_len)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        need_weights: bool = False,
        *,
        verbose: bool = False,
        log_stream: Optional[TextIO] = None,
    ):
        """
        query/key/value: (batch_size, seq_len, d_model)
        attn_mask: boolean mask with shape broadcastable to (batch_size, num_heads, query_len, key_len).
                   True = mask out (set to -inf before softmax)
        """
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        # Project embeddings to Q/K/V spaces, then split into heads.
        projected_q = self.W_q(query)
        projected_k = self.W_k(key)
        projected_v = self.W_v(value)

        Q = self._split_heads(projected_q)   # (batch_size, num_heads, query_len, head_dim)
        K = self._split_heads(projected_k)   # (batch_size, num_heads, key_len, head_dim)
        V = self._split_heads(projected_v)   # (batch_size, num_heads, key_len, head_dim)

        if verbose and log_stream is not None:
            _log_header(log_stream, "Multi-Head Attention: Projections")
            _log_tensor(log_stream, "Input query", query)
            _log_tensor(log_stream, "Input key", key)
            _log_tensor(log_stream, "Input value", value)
            _log_tensor(log_stream, "Projected Q", projected_q)
            _log_tensor(log_stream, "Projected K", projected_k)
            _log_tensor(log_stream, "Projected V", projected_v)
            _log_tensor(log_stream, "Split Q", Q)
            _log_tensor(log_stream, "Split K", K)
            _log_tensor(log_stream, "Split V", V)

        # Prepare mask
        if attn_mask is not None:
            # Expect (batch_size, 1, query_len, key_len) or (1, 1, query_len, key_len). If (query_len, key_len), expand.
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,query_len,key_len)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)               # (batch_size,1,query_len,key_len)
            # else assume shape already broadcastable

        if verbose and log_stream is not None and attn_mask is not None:
            _log_mask(log_stream, "Attention mask", attn_mask)

        # Attention
        context, attn_weights = self.attn(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            verbose=verbose,
            log_stream=log_stream,
        )

        # Merge heads and project back to model dimensionality.
        context = self._merge_heads(context)     # (batch_size, query_len, d_model)
        out = self.out(self.dropout(context))    # (batch_size, query_len, d_model)

        if verbose and log_stream is not None:
            _log_header(log_stream, "Multi-Head Attention: Output")
            _log_tensor(log_stream, "Merged context", context)
            _log_tensor(log_stream, "Final output", out)

        if need_weights:
            return out, attn_weights             # attn_weights: (batch_size, num_heads, query_len, key_len)
        return out


# ---------------------------------------------------------------------------
# Small demo showcasing the layer in action.
# ---------------------------------------------------------------------------

# Batch size, sequence length, hidden dimension, and number of heads.
batch_size, seq_len, model_dim, num_heads = 2, 5, 32, 4
# Fake token embeddings.
token_embeddings = torch.randn(batch_size, seq_len, model_dim)

# Initialise the attention module.
mha = MultiHeadAttention(d_model=model_dim, num_heads=num_heads, dropout=0.1)

# Prepare log file capturing every intermediate tensor.
log_path = Path(__file__).with_name("torch_attention_log.txt")
with log_path.open("w", encoding="utf-8") as log_file:
    _log_header(log_file, "Input Embeddings")
    _log_tensor(log_file, "Token embeddings", token_embeddings)

    # 1) No mask
    y, w = mha(
        token_embeddings,
        token_embeddings,
        token_embeddings,
        need_weights=True,
        verbose=True,
        log_stream=log_file,
    )
    print("y shape:", y.shape)              # (batch_size, seq_len, model_dim)
    print("attn weights shape:", w.shape)   # (batch_size, num_heads, seq_len, seq_len)

    # 2) Padding mask example: mask out the last 2 keys for sample 0
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[0, -2:] = True  # True means "MASK this position"
    _log_header(log_file, "Padding Mask Setup")
    _log_mask(log_file, "Key padding mask", key_padding_mask)

    # Broadcast to (batch_size, 1, 1, seq_len)
    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

    y_pad = mha(
        token_embeddings,
        token_embeddings,
        token_embeddings,
        attn_mask=attn_mask,
        verbose=True,
        log_stream=log_file,
    )
    print("y_pad shape:", y_pad.shape)

    # 3) Causal mask (decoder self-attention)
    causal = MultiHeadAttention.make_causal_mask(seq_len, device=token_embeddings.device)  # (seq_len, seq_len), True above diagonal
    _log_header(log_file, "Causal Mask Setup")
    _log_mask(log_file, "Causal mask", causal)

    y_causal = mha(
        token_embeddings,
        token_embeddings,
        token_embeddings,
        attn_mask=causal,
        verbose=True,
        log_stream=log_file,
    )
    print("y_causal shape:", y_causal.shape)
