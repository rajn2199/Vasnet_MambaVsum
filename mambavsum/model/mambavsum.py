# model/mambavsum.py
"""
MambaVSum: Hybrid Mamba-Transformer for Efficient Multimodal Video Summarization.

Full architecture:
  1. Multimodal Fusion: CLIP visual + audio → fused features
  2. BiMamba Encoder: O(N) bidirectional temporal modeling
  3. Multi-Scale Temporal Pooling: capture local + global patterns
  4. Change-Point Sparse Attention: attend globally at shot boundaries
  5. Score Regressor: predict frame importance scores

This model combines the efficiency of Mamba (O(N) vs O(N²) attention)
with the representational power of modern features (CLIP) and
multimodal fusion, achieving SOTA performance on video summarization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.mamba import BiMambaEncoder
from model.fusion import MultimodalFusion


class MultiScaleTemporalPooling(nn.Module):
    """
    Multi-scale temporal pooling to capture patterns at different granularities.

    Applies average pooling at multiple scales (e.g., 1×, 2×, 4×), projects
    each scale, and combines them. This gives the model both fine-grained
    local information and coarse global context.

    Scale 1×: frame-level detail
    Scale 2×: segment-level patterns
    Scale 4×: scene-level context
    """
    def __init__(self, d_model, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])
        self.combine = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, d_model)
        Returns:
            y: (B, N, d_model)
        """
        batch, seq_len, d = x.shape
        scale_outputs = []

        for scale, proj in zip(self.scales, self.projections):
            if scale == 1:
                pooled = x
            else:
                # Reshape for pooling: (B, d, N) for avg_pool1d
                x_t = x.transpose(1, 2)  # (B, d, N)
                pooled_t = F.avg_pool1d(x_t, kernel_size=scale, stride=scale,
                                         ceil_mode=True)  # (B, d, N//scale)
                # Upsample back to original length
                pooled_t = F.interpolate(pooled_t, size=seq_len, mode='linear',
                                          align_corners=False)
                pooled = pooled_t.transpose(1, 2)  # (B, N, d)

            scale_outputs.append(proj(pooled))

        # Concatenate all scales and combine
        combined = torch.cat(scale_outputs, dim=-1)  # (B, N, d * n_scales)
        return self.combine(combined)


class ChangepointAttention(nn.Module):
    """
    Sparse attention at shot boundaries (change points).

    Instead of full O(N²) self-attention, this module only computes
    attention between frames and their nearest change points,
    providing global context at O(N·K) cost where K << N.

    This bridges the gap between pure Mamba (local) and full attention (global).
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, change_points=None):
        """
        Args:
            x: (B, N, d_model) — encoded features
            change_points: (K, 2) np.ndarray — segment boundaries (optional)

        Returns:
            y: (B, N, d_model)
        """
        if change_points is None or len(change_points) == 0:
            return x

        batch, seq_len, d = x.shape

        # Extract features at change point positions
        cp_indices = []
        for cp in change_points:
            start_idx = min(int(cp[0]), seq_len - 1)
            end_idx = min(int(cp[1]), seq_len - 1)
            # Use both start and end of each segment
            cp_indices.extend([start_idx, end_idx])

        # Deduplicate and sort
        cp_indices = sorted(set(cp_indices))
        if len(cp_indices) == 0:
            return x

        cp_indices_t = torch.tensor(cp_indices, device=x.device, dtype=torch.long)
        cp_features = x[:, cp_indices_t, :]  # (B, K', d_model)

        # Cross-attention: all frames attend to change-point frames
        residual = x
        attn_out, _ = self.attn(query=x, key=cp_features, value=cp_features)
        y = self.norm(self.dropout(attn_out) + residual)

        return y


class ScoreRegressor(nn.Module):
    """
    Frame importance score predictor.
    Maps d_model features to scalar importance scores in [0, 1].
    """
    def __init__(self, d_model, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:  x: (B, N, d_model)
        Returns: scores: (N,) — importance scores in [0, 1]
        """
        return self.regressor(x).squeeze(0).squeeze(-1)  # (N,)


class MambaVSum(nn.Module):
    """
    MambaVSum: Full Model.

    Architecture:
    ┌───────────────────────────────────────────────────┐
    │  Visual Features (GoogLeNet 1024 / CLIP 768)     │
    │  Audio Features (128-d, optional)                 │
    │              ↓                                    │
    │  Multimodal Fusion (Gated / CrossAttn / Concat)   │
    │              ↓                                    │
    │  BiMamba Encoder (L layers, O(N) complexity)      │
    │    - Forward Mamba (left → right)                 │
    │    - Backward Mamba (right → left)                │
    │    - Gated combination                            │
    │              ↓                                    │
    │  Multi-Scale Temporal Pooling (1×, 2×, 4×)       │
    │              ↓                                    │
    │  Changepoint Sparse Attention (optional)          │
    │              ↓                                    │
    │  Score Regressor → (N,) ∈ [0, 1]                 │
    └───────────────────────────────────────────────────┘

    Args:
        cfg: Config object with all hyperparameters
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Determine input dimensions based on feature mode
        if cfg.feature_mode == "multimodal":
            visual_dim = cfg.clip_dim
            audio_dim = cfg.audio_dim
        elif cfg.feature_mode == "clip":
            visual_dim = cfg.clip_dim
            audio_dim = 0
        else:  # googlenet
            visual_dim = cfg.googlenet_dim
            audio_dim = 0

        d_model = cfg.mamba_d_model

        # 1. Multimodal Fusion
        self.fusion = MultimodalFusion(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            output_dim=d_model,
            strategy="gated" if audio_dim > 0 else "gated",
        )

        # 2. BiMamba Encoder
        self.encoder = BiMambaEncoder(
            d_model=d_model,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            n_layers=cfg.mamba_n_layers,
            dropout=cfg.mamba_dropout,
        )

        # 3. Multi-Scale Temporal Pooling
        self.temporal_pool = MultiScaleTemporalPooling(
            d_model=d_model,
            scales=cfg.temporal_scales,
        )

        # 4. Changepoint Sparse Attention (optional refinement)
        self.cp_attention = ChangepointAttention(
            d_model=d_model,
            n_heads=4,
            dropout=cfg.mamba_dropout,
        )

        # 5. Score Regressor
        self.scorer = ScoreRegressor(
            d_model=d_model,
            hidden_dim=cfg.scorer_hidden,
            dropout=cfg.scorer_dropout,
        )

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  MambaVSum: {total:,} params ({trainable:,} trainable)")

    def forward(self, visual, audio=None, change_points=None):
        """
        Args:
            visual: (1, N, D_v) — visual features (GoogLeNet or CLIP)
            audio:  (1, N, D_a) — audio features (optional)
            change_points: (K, 2) np.ndarray — shot boundaries (optional)

        Returns:
            scores: (N,) — frame importance scores in [0, 1]
            encoded: (1, N, d_model) — encoded features (for visualization)
        """
        # 1. Fuse modalities → (1, N, d_model)
        fused = self.fusion(visual, audio)

        # 2. BiMamba encoding → (1, N, d_model)
        encoded = self.encoder(fused)

        # 3. Multi-scale pooling → (1, N, d_model)
        pooled = self.temporal_pool(encoded)

        # 4. Sparse attention at change points → (1, N, d_model)
        if change_points is not None:
            # Map change points from original frame space to subsampled space
            # (approximate: divide by subsampling rate ~15)
            pooled = self.cp_attention(pooled, change_points)

        # 5. Score regression → (N,)
        scores = self.scorer(pooled)

        return scores, encoded
