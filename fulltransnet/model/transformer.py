# model/transformer.py
"""
FullTransNet: Full Transformer with Local-Global Attention for Video Summarization.
Encoder-Decoder architecture following the paper by Lan et al. (2024).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from model.attention import LocalGlobalMultiHeadAttention, MultiHeadAttention


class FixedPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, dim_mid, max_length=None):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_length, dim_mid)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_mid, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / dim_mid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, dim_mid) or (seq_len, dim_mid)"""
        if x.dim() == 3:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + Variable(self.pe[:, :x.size(0)], requires_grad=False)
        return x


class VideoEmbedding(nn.Module):
    """Linear embedding layer for video features: D_in -> D_mid."""

    def __init__(self, dim_in, dim_mid):
        super(VideoEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )

    def forward(self, x):
        return self.embedding(x)


class PoswiseFeedForwardNet(nn.Module):
    """Position-wise Feed-Forward Network with residual + LayerNorm."""

    def __init__(self, dim_mid=64, dff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = dim_mid
        self.d_ff = dff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = output + residual
        output = self.layer_norm(output)
        return output


# ─── Mask utilities ────────────────────────────────────────────────

def get_attn_pad_mask_video_de(seq_q, seq_k):
    """Decoder padding mask — all zeros (no padding in decoder)."""
    len_q = seq_q.size(0)
    len_k = seq_k.size(0)
    pad_attn_mask = torch.zeros((1, len_q, len_k), dtype=torch.bool)
    return pad_attn_mask


def get_attn_pad_mask_video(T, seq_q, seq_k, max_length, global_idx):
    """
    Build encoder attention mask for Local-Global Attention.
    Returns: (1, max_length) where:
       1 = real token with local attention
      -1 = global token (attends to/from all)
       0 = padding
    """
    device = seq_q.device
    mask = torch.cat(
        (torch.ones(T), torch.zeros(max_length - T)), dim=-1
    ).to(device)

    # Mark global positions (from change points, downsampled by 15)
    global_idx_ds = torch.tensor(global_idx // 15).to(device)
    # Clamp indices to valid range
    global_idx_ds = global_idx_ds[global_idx_ds < max_length]
    global_idx_ds = global_idx_ds[global_idx_ds >= 0]
    mask[global_idx_ds] = -1

    mask = mask.unsqueeze(0)
    return mask


def get_attn_subsequent_mask(seq):
    """Causal (upper-triangular) mask for decoder self-attention."""
    seq = seq.unsqueeze(0)
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


# ─── Encoder / Decoder Layers ─────────────────────────────────────

class SparseEncoderLayer(nn.Module):
    """Encoder layer: Local-Global MHA → Feed-Forward."""

    def __init__(self, window_size, dim_mid, dff):
        super(SparseEncoderLayer, self).__init__()
        self.enc_self_attn = LocalGlobalMultiHeadAttention(
            dim_mid=dim_mid,
            window_size=window_size
        )
        self.pos_ffn = PoswiseFeedForwardNet(dim_mid=dim_mid, dff=dff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    """Decoder layer: Self-Attention → Cross-Attention → Feed-Forward."""

    def __init__(self, dim_mid, heads, dff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(dim_mid=dim_mid, heads=heads)
        self.dec_enc_attn = MultiHeadAttention(dim_mid=dim_mid, heads=heads)
        self.pos_ffn = PoswiseFeedForwardNet(dim_mid=dim_mid, dff=dff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn, _ = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        dec_outputs, dec_enc_attn, context = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn, context


# ─── Encoder / Decoder ────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Encoder with N stacked SparseEncoderLayers using Local-Global Attention.
    """

    def __init__(self, dim_in, dim_mid, enlayers, length, window_size, dff):
        super(Encoder, self).__init__()
        self.src_emb = VideoEmbedding(dim_in, dim_mid)
        self.max_length = length
        self.nlayers = enlayers
        self.position_embeddings = FixedPositionalEncoding(dim_mid, max_length=length)

        self.local_layers = nn.ModuleList([
            SparseEncoderLayer(window_size, dim_mid, dff)
            for _ in range(self.nlayers)
        ])

    def forward(self, T, enc_inputs, global_idx):
        """
        T: actual number of frames (before padding)
        enc_inputs: (batch, max_length, dim_mid) — already embedded
        global_idx: array of global attention positions (frame indices)
        """
        enc_outputs = enc_inputs + self.position_embeddings(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask_video(
            T, enc_inputs.squeeze(0), enc_inputs.squeeze(0),
            self.max_length, global_idx
        )

        enc_self_attns = []
        for layer in self.local_layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs.squeeze(0), enc_self_attns


class Decoder(nn.Module):
    """
    Decoder with N stacked DecoderLayers using standard Multi-Head Attention.
    """

    def __init__(self, dim_mid, delayers, length, heads, dff):
        super(Decoder, self).__init__()
        self.d_model = dim_mid
        self.delayers = delayers
        self.position_embeddings = FixedPositionalEncoding(dim_mid, max_length=length)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_mid, heads, dff)
            for _ in range(self.delayers)
        ])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: (seq_len, dim_mid) — target key frame features
        enc_inputs: (batch, max_length, dim_mid)
        enc_outputs: (max_length, dim_mid) — encoder output
        """
        dec_outputs = dec_inputs + self.position_embeddings(dec_inputs)

        dec_self_attn_pad_mask = get_attn_pad_mask_video_de(
            dec_inputs, dec_inputs
        ).to(dec_inputs.device)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(
            dec_inputs
        ).to(dec_inputs.device)
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0
        )

        dec_enc_attn_mask = get_attn_pad_mask_video_de(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn, context = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


# ─── Full Transformer Model ───────────────────────────────────────

class Transformer(nn.Module):
    """
    FullTransNet: Full Transformer with Local-Global Attention.

    Architecture:
    1. VideoEmbedding: project 1024-dim features to dim_mid
    2. Encoder: N layers of Local-Global Attention + FFN
    3. Decoder: M layers of Self-Attention + Cross-Attention + FFN
    4. Projection: output importance scores via softmax
    """

    def __init__(self, T, dim_in, heads, enlayers, delayers, dim_mid,
                 length, window_size, stride, dff):
        super(Transformer, self).__init__()

        self.dim_mid = dim_mid
        self.d_k = dim_mid / heads
        self.d_v = dim_mid / heads
        self.n_heads = heads
        self.length = length

        self.embedding = VideoEmbedding(dim_in, dim_mid)
        self.encoder = Encoder(dim_in, dim_mid, enlayers, length, window_size, dff)
        self.decoder = Decoder(dim_mid, delayers, length, heads, dff)
        self.projection = nn.Linear(self.dim_mid, self.length, bias=False)

    def forward(self, x, target, global_idx):
        """
        x: (1, T, 1024) — video frame features
        target: (n_keyframes, 1024) — key frame features for decoder input
        global_idx: array of change point frame indices
        Returns: (n_keyframes, T) softmax scores, encoder/decoder attention maps
        """
        T = x.shape[1]
        device = x.device

        # Pad input to max length
        pad = torch.zeros((1, self.length - T, 1024), device=device)
        x = torch.cat((x, pad), dim=1)

        # Embed input and target
        enc_inputs = self.embedding(x)        # (1, length, dim_mid)
        dec_inputs = self.embedding(target)    # (n_keyframes, dim_mid)

        # Encode
        enc_outputs, enc_self_attns = self.encoder(T, enc_inputs, global_idx)

        # Decode
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs
        )

        # Project to score space
        dec_logits = self.projection(dec_outputs)
        dec_logits = dec_logits.view(-1, dec_logits.size(-1))
        dec_logits = dec_logits[:, 0:T]
        generator_dec_output = F.softmax(dec_logits, dim=-1)

        return generator_dec_output, enc_self_attns, dec_self_attns, dec_enc_attns
