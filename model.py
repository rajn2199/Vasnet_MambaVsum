# model.py
"""
VASNet: Video Attention-based Summarization Network
Exact implementation following the paper (Fajtl et al., 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class SelfAttention(nn.Module):
    """
    Multiplicative soft self-attention — Eq. 1-5 in the paper.
    Single-head attention as specified in the original VASNet.
    """
    def __init__(self, input_size: int = 1024, scale: float = 0.06, dropout: float = 0.5):
        super().__init__()
        self.scale = scale
        # Learnable projection matrices U, V, C — each (D × D)
        self.U = nn.Linear(input_size, input_size, bias=False)
        self.V = nn.Linear(input_size, input_size, bias=False)
        self.C = nn.Linear(input_size, input_size, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.U, self.V, self.C]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x      : (N, D)
        returns: context (N, D), alpha (N, N)
        """
        UX = self.U(x)  # (N, D)
        VX = self.V(x)  # (N, D)

        # Eq. 1 — e_{t,i} = scale * (Ux_i)^T (Vx_t)
        e = self.scale * torch.matmul(VX, UX.T)  # (N, N)

        # Eq. 3 — softmax → attention probabilities
        alpha = F.softmax(e, dim=1)  # (N, N)
        alpha = self.attn_drop(alpha)

        # Eq. 4-5 — B = Cx,  c_t = Σ α_{t,i} · b_i
        B = self.C(x)  # (N, D)
        context = torch.matmul(alpha, B)  # (N, D)

        return context, alpha


class VASNet(nn.Module):
    """
    VASNet (Fig. 2 in paper) - Exact architecture.
    Input  : (1, N, D)  — one complete video
    Output : (N,) scores ∈ [0, 1],  (N, N) attention weights
    """
    def __init__(self, cfg: Config):
        super().__init__()

        D = cfg.input_size
        H = cfg.hidden_size

        # ── Attention block (Eq. 1-5) ────────────────────────
        self.attention = SelfAttention(D, cfg.attn_scale, cfg.dropout)
        
        # Eq. 6 — Linear projection W for residual
        self.W = nn.Linear(D, D, bias=False)
        self.drop1 = nn.Dropout(p=cfg.dropout)
        self.norm1 = nn.LayerNorm(D)

        # ── Regressor (Eq. 7) — 2-layer MLP ──────────────────
        self.fc1 = nn.Linear(D, H // 2)
        self.drop2 = nn.Dropout(p=cfg.dropout)
        self.norm2 = nn.LayerNorm(H // 2)
        self.fc2 = nn.Linear(H // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.squeeze(0)  # (1,N,D) → (N,D)

        context, attn = self.attention(x)  # (N,D), (N,N)

        # Eq. 6 — residual + dropout + LayerNorm
        k = self.W(context) + x  # residual sum
        k = self.drop1(k)
        k = self.norm1(k)  # (N, D)

        # 2-layer MLP regressor
        y = F.relu(self.fc1(k))  # (N, H/2)
        y = self.drop2(y)
        y = self.norm2(y)
        y = torch.sigmoid(self.fc2(y))  # (N, 1)

        return y.squeeze(1), attn  # (N,), (N,N)