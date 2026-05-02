# model/fusion.py
"""
Multimodal Fusion Module for MambaVSum.
Fuses visual (CLIP) and audio features into a unified representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU).
    h_v = tanh(W_v · v), h_a = tanh(W_a · a)
    z = σ(W_z · [v; a])  — learned gate
    f = z ⊙ h_v + (1-z) ⊙ h_a
    Ref: Arevalo et al., "Gated Multimodal Units" (2017)
    """
    def __init__(self, visual_dim, audio_dim, output_dim):
        super().__init__()
        self.visual_proj = nn.Sequential(nn.Linear(visual_dim, output_dim), nn.Tanh())
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, output_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(visual_dim + audio_dim, output_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual, audio):
        h_v = self.visual_proj(visual)
        h_a = self.audio_proj(audio)
        z = self.gate(torch.cat([visual, audio], dim=-1))
        return self.layer_norm(z * h_v + (1 - z) * h_a)


class CrossModalAttention(nn.Module):
    """Cross-modal attention: visual attends to audio and vice versa."""
    def __init__(self, visual_dim, audio_dim, output_dim, n_heads=4):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.audio_proj = nn.Linear(audio_dim, output_dim)
        self.v2a_attn = nn.MultiheadAttention(output_dim, n_heads, dropout=0.1, batch_first=True)
        self.a2v_attn = nn.MultiheadAttention(output_dim, n_heads, dropout=0.1, batch_first=True)
        self.out_proj = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.ReLU(), nn.LayerNorm(output_dim))

    def forward(self, visual, audio):
        v, a = self.visual_proj(visual), self.audio_proj(audio)
        v2a, _ = self.v2a_attn(query=v, key=a, value=a)
        a2v, _ = self.a2v_attn(query=a, key=v, value=v)
        return self.out_proj(torch.cat([v2a, a2v], dim=-1))


class MultimodalFusion(nn.Module):
    """
    Top-level fusion module. Strategies: "gated", "cross_attention", "concat".
    Handles single-modality (visual only) with a linear projection.
    """
    def __init__(self, visual_dim, audio_dim=0, output_dim=256, strategy="gated", n_heads=4):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.has_audio = audio_dim > 0

        if self.has_audio:
            if strategy == "gated":
                self.fusion = GatedFusionUnit(visual_dim, audio_dim, output_dim)
            elif strategy == "cross_attention":
                self.fusion = CrossModalAttention(visual_dim, audio_dim, output_dim, n_heads)
            elif strategy == "concat":
                self.fusion = nn.Sequential(
                    nn.Linear(visual_dim + audio_dim, output_dim), nn.ReLU(),
                    nn.Dropout(0.1), nn.LayerNorm(output_dim))
            else:
                raise ValueError(f"Unknown fusion strategy: {strategy}")
        else:
            self.fusion = nn.Sequential(
                nn.Linear(visual_dim, output_dim), nn.ReLU(),
                nn.Dropout(0.1), nn.LayerNorm(output_dim))

    def forward(self, visual, audio=None):
        if self.has_audio and audio is not None:
            return self.fusion(visual, audio)
        elif isinstance(self.fusion, nn.Sequential):
            return self.fusion(visual)
        else:
            dummy = torch.zeros(*visual.shape[:2], self.audio_dim, device=visual.device)
            return self.fusion(visual, dummy)
