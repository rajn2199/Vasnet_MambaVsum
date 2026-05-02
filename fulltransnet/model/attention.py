# model/attention.py
"""
Pure PyTorch implementation of Local-Global Attention for FullTransNet.
Replaces the Longformer-based sliding window attention from the original code
with a mask-based approach that works on any platform (including Windows/CPU).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention used in the decoder."""

    def __init__(self, dim_mid=64, heads=8):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_mid = dim_mid
        self.d_k = dim_mid // heads

    def forward(self, Q, K, V, attn_mask):
        """
        Q, K, V: (batch, heads, seq_len, d_k)
        attn_mask: (1, seq_len, seq_len) bool — True means masked
        """
        attn_mask = attn_mask.to(torch.bool)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask.to(scores.device), -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class LocalGlobalAttention(nn.Module):
    """
    Local-Global Scaled Dot-Product Attention — pure PyTorch.

    Instead of using Longformer's custom CUDA kernels (sliding_chunks),
    we implement the same semantics using a band mask:
      - Each query token attends to tokens within [pos - window, pos + window]
      - Global tokens (from change-point positions) attend to ALL tokens
      - Padding tokens are masked out

    This produces mathematically identical results to the Longformer variant
    but runs on any device (CPU/CUDA, any OS).
    """

    def __init__(self, window_size=16, dim_mid=64, heads=8):
        super(LocalGlobalAttention, self).__init__()
        self.window_size = window_size
        self.dim_mid = dim_mid
        self.num_heads = heads
        self.head_dim = dim_mid // heads
        self.dropout = nn.Dropout(0.1)

        # Local attention projections
        self.W_Q = nn.Linear(dim_mid, dim_mid)
        self.W_K = nn.Linear(dim_mid, dim_mid)
        self.W_V = nn.Linear(dim_mid, dim_mid)

        # Global attention projections (separate params for global tokens)
        self.W_Q_global = nn.Linear(dim_mid, dim_mid)
        self.W_K_global = nn.Linear(dim_mid, dim_mid)
        self.W_V_global = nn.Linear(dim_mid, dim_mid)

    def _build_local_mask(self, seq_len, device):
        """Build a band mask: True = allowed to attend, False = blocked."""
        # Create position indices
        rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        # Band: allow attending within window_size distance
        mask = (cols - rows).abs() <= self.window_size  # (seq_len, seq_len)
        return mask

    def forward(self, hidden_states, attn_mask):
        """
        hidden_states: (batch=1, seq_len, dim_mid  )
        attn_mask: (1, max_length) where:
            1 = real token (local attention)
           -1 = global token (attends to/from all)
            0 = padding (masked out)
        Returns: (batch, seq_len, dim_mid), attention_weights
        """
        bsz, seq_len, embed_dim = hidden_states.size()
        device = hidden_states.device

        # Identify token types from the mask
        mask_squeezed = attn_mask.squeeze(0)  # (seq_len,)
        is_padding = (mask_squeezed == 0)     # padding positions
        is_global = (mask_squeezed == -1)     # global attention positions
        is_real = ~is_padding                 # all non-padding (real + global)

        # --- Project Q, K, V ---
        q = self.W_Q(hidden_states)  # (bsz, seq_len, dim_mid)
        k = self.W_K(hidden_states)
        v = self.W_V(hidden_states)

        # Reshape to multi-head: (bsz, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = q / math.sqrt(self.head_dim)

        # --- Build attention mask ---
        # Start with local band mask
        attn_allowed = self._build_local_mask(seq_len, device)  # (seq_len, seq_len)

        # Global tokens: they attend to ALL real tokens, and ALL tokens attend to them
        global_mask = is_global.unsqueeze(0).expand(seq_len, -1)  # rows where col is global
        global_mask_t = is_global.unsqueeze(1).expand(-1, seq_len)  # cols where row is global
        attn_allowed = attn_allowed | global_mask | global_mask_t

        # Mask out padding in both dimensions
        padding_mask_col = is_padding.unsqueeze(0).expand(seq_len, -1)  # can't attend TO padding
        padding_mask_row = is_padding.unsqueeze(1).expand(-1, seq_len)  # padding can't attend
        attn_allowed = attn_allowed & ~padding_mask_col & ~padding_mask_row

        # Convert to additive mask: 0 for allowed, -inf for blocked
        # (1, 1, seq_len, seq_len) for broadcasting over batch and heads
        attn_bias = torch.zeros(seq_len, seq_len, device=device)
        attn_bias.masked_fill_(~attn_allowed, -1e4)
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # --- Compute local attention scores ---
        scores = torch.matmul(q, k.transpose(-1, -2))  # (bsz, heads, seq_len, seq_len)
        scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        # Zero out attention from/to padding
        attn_weights = attn_weights.masked_fill(
            is_padding.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(attn_weights), 0.0
        )
        attn_probs = self.dropout(attn_weights.to(hidden_states.dtype))

        # --- Compute context ---
        context = torch.matmul(attn_probs, v)  # (bsz, heads, seq_len, head_dim)

        # --- Override global token outputs with full global attention ---
        if is_global.any():
            global_indices = is_global.nonzero(as_tuple=True)[0]  # positions of global tokens

            # Use separate global projections for global tokens
            q_global = self.W_Q_global(hidden_states[:, global_indices, :])  # (bsz, n_global, dim_mid)
            k_global = self.W_K_global(hidden_states)  # (bsz, seq_len, dim_mid)
            v_global = self.W_V_global(hidden_states)  # (bsz, seq_len, dim_mid)

            q_global = q_global.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k_global = k_global.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_global = v_global.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            q_global = q_global / math.sqrt(self.head_dim)

            # Full attention for global queries
            global_scores = torch.matmul(q_global, k_global.transpose(-1, -2))
            # Mask padding columns
            pad_mask = is_padding.unsqueeze(0).unsqueeze(0).unsqueeze(1)  # (1, 1, 1, seq_len)
            global_scores = global_scores.masked_fill(pad_mask.expand_as(global_scores), -1e4)

            global_attn = F.softmax(global_scores, dim=-1, dtype=torch.float32)
            global_attn = self.dropout(global_attn.to(hidden_states.dtype))
            global_context = torch.matmul(global_attn, v_global)  # (bsz, heads, n_global, head_dim)

            # Replace context at global positions
            context[:, :, global_indices, :] = global_context

        # Reshape back: (bsz, seq_len, dim_mid)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)

        # Store weights for visualization (detached)
        attn_weights_np = attn_weights.detach().cpu().numpy()

        return context, attn_weights_np


class LocalGlobalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention wrapper using Local-Global Attention for the encoder.
    Includes residual connection + LayerNorm.
    """

    def __init__(self, dim_mid=64, heads=8, window_size=16):
        super(LocalGlobalMultiHeadAttention, self).__init__()
        self.dim_mid = dim_mid
        self.n_heads = heads
        self.window_size = window_size

        self.lga = LocalGlobalAttention(
            window_size=window_size,
            dim_mid=dim_mid,
            heads=heads
        )
        self.linear = nn.Linear(dim_mid, dim_mid)
        self.layer_norm = nn.LayerNorm(dim_mid)

    def forward(self, Q, K, V, attn_mask):
        """
        Q, K, V: (batch, seq_len, dim_mid) — for self-attention Q=K=V
        attn_mask: (1, seq_len) with values {0, 1, -1}
        """
        residual = Q
        context, attn = self.lga(Q, attn_mask)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention for the decoder."""

    def __init__(self, dim_mid=64, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.dim_mid = dim_mid
        self.d_k = dim_mid // heads
        self.d_v = dim_mid // heads
        self.n_heads = heads

        self.W_Q = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_K = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_V = nn.Linear(dim_mid, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, dim_mid)
        self.layer_norm = nn.LayerNorm(dim_mid)

    def forward(self, Q, K, V, attn_mask):
        """
        Q, K, V: (seq_len, batch, dim_mid)  — note: seq_len first for decoder
        attn_mask: (1, seq_len, seq_len) bool
        """
        residual, batch_size, seq_len = Q, Q.size(0), Q.size(1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(
            dim_mid=self.dim_mid, heads=self.n_heads
        )(q_s, k_s, v_s, attn_mask)

        context = context.squeeze(0).transpose(0, 1).contiguous().view(
            seq_len, -1, self.n_heads * self.d_v
        )
        output = self.linear(context).transpose(0, 1)

        return self.layer_norm(output + residual), attn, context
