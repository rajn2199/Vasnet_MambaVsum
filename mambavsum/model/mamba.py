# model/mamba.py
"""
Bidirectional Mamba (Selective State Space Model) for Video Summarization.

Pure PyTorch implementation of the Mamba selective scan — no custom CUDA
kernels required. This enables running on any device (CPU/GPU, any OS).

Key ideas:
  - Selective State Space Model (S6): input-dependent state transitions
  - Bidirectional scanning: forward + backward for full temporal context
  - Hardware-friendly: uses parallel scan via associative scan trick
  - O(N) complexity vs O(N²) for self-attention

Reference:
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  (COLM 2024)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — the core of Mamba.

    Unlike classical SSMs with fixed A, B, C matrices, the selective SSM
    makes B, C, and Δ (discretization step) input-dependent, allowing
    the model to selectively remember or forget information.

    For a sequence x ∈ R^(L, D):
      - Δ_t = softplus(Linear(x_t))        — step size (input-dependent)
      - B_t = Linear(x_t)                   — input matrix (input-dependent)
      - C_t = Linear(x_t)                   — output matrix (input-dependent)
      - A   = fixed negative log-spaced      — state matrix (learned but structured)

    Discretization (ZOH):
      Ā_t = exp(Δ_t · A)
      B̄_t = Δ_t · B_t

    Recurrence:
      h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t
      y_t = C_t · h_t
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # ── Input projection: d_model -> 2 * d_inner (for x and z gate) ──
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # ── 1D convolution (local context before SSM) ────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise conv
            bias=True,
        )

        # ── SSM parameters ───────────────────────────────────────────────
        # Δ (delta) projection: d_inner -> d_inner
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Initialize dt bias for softplus to give reasonable step sizes
        dt_init_std = (1.0 / d_model) ** 0.5
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)

        # B projection: d_inner -> d_state
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # C projection: d_inner -> d_state
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # A: fixed structure, learned log-spaced values
        # A ∈ R^(d_inner, d_state) — initialized as negative log-spaced
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))  # Store as log for stability

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ── Output projection: d_inner -> d_model ───────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _selective_scan(self, x, dt, B, C, D, chunk_size=32):
        """
        Vectorized selective scan using chunked cumulative products.

        Instead of iterating over every timestep in Python (O(L) iterations),
        this processes chunks of `chunk_size` frames at once using vectorized
        torch.cumprod and torch.cumsum, reducing to O(L/chunk_size) iterations.

        Math: For recurrence h_t = dA_t * h_{t-1} + dBx_t, within a chunk:
          h_t = cum_dA[t] * (h_prev + cumsum(dBx[k] / cum_dA[k]))
        where cum_dA[t] = cumprod of dA from start of chunk to t.

        Args:
            x:  (B, L, d_inner)  — input after conv
            dt: (B, L, d_inner)  — discretization step sizes
            B:  (B, L, d_state)  — input matrices
            C:  (B, L, d_state)  — output matrices
            D:  (d_inner,)       — skip connection

        Returns:
            y:  (B, L, d_inner)  — output sequence
        """
        batch, seq_len, d_inner = x.shape
        d_state = B.shape[-1]

        # Get A from log parameterization
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Precompute all discretized parameters at once (fully vectorized)
        dt_exp = dt.unsqueeze(-1)                       # (B, L, d_inner, 1)
        A_exp = A.unsqueeze(0).unsqueeze(0)             # (1, 1, d_inner, d_state)

        dA = torch.exp(dt_exp * A_exp)                  # (B, L, d_inner, d_state)
        dBx = dt_exp * B.unsqueeze(2) * x.unsqueeze(-1) # (B, L, d_inner, d_state)

        # Chunked vectorized scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        y_chunks = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)

            dA_c  = dA[:, start:end]    # (B, C, d_inner, d_state)
            dBx_c = dBx[:, start:end]   # (B, C, d_inner, d_state)
            C_c   = C[:, start:end]     # (B, C, d_state)

            # Cumulative product of dA within this chunk
            cum_dA = torch.cumprod(dA_c, dim=1)          # (B, C, d_inner, d_state)

            # h_t = cum_dA[t] * (h_prev + cumsum(dBx[k] / cum_dA[k]))
            ratio = dBx_c / (cum_dA + 1e-6)
            cum_ratio = torch.cumsum(ratio, dim=1)

            h_all = cum_dA * (h.unsqueeze(1) + cum_ratio) # (B, C, d_inner, d_state)

            # Output: y_t = sum(h_t * C_t, dim=d_state)
            y_c = torch.sum(h_all * C_c.unsqueeze(2), dim=-1)  # (B, C, d_inner)
            y_chunks.append(y_c)

            # Carry state to next chunk
            h = h_all[:, -1]                              # (B, d_inner, d_state)

        y = torch.cat(y_chunks, dim=1)  # (B, L, d_inner)

        # Skip connection
        y = y + D.unsqueeze(0).unsqueeze(0) * x

        return y

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model) — input sequence

        Returns:
            y: (B, L, d_model) — output sequence
        """
        batch, seq_len, _ = x.shape

        # ── Input projection + gate ──────────────────────────
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # ── 1D convolution for local context ─────────────────
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # ── Compute input-dependent SSM parameters ───────────
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)
        B = self.B_proj(x_conv)  # (B, L, d_state)
        C = self.C_proj(x_conv)  # (B, L, d_state)

        # ── Run selective scan ───────────────────────────────
        y = self._selective_scan(x_conv, dt, B, C, self.D)

        # ── Gated output (SiLU gate from z) ──────────────────
        y = y * F.silu(z)

        # ── Output projection ────────────────────────────────
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block.

    Runs two independent Mamba SSMs:
      - Forward:  scans left → right (captures past context)
      - Backward: scans right → left (captures future context)

    Outputs are combined via learned gating:
      y = gate * y_fwd + (1 - gate) * y_bwd

    This is critical for video summarization because importance of a frame
    depends on BOTH what came before AND what comes after.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()

        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)

        # Learned gating: decides how much forward vs backward to use
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)

        Returns:
            y: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # Forward scan: left → right
        y_fwd = self.forward_ssm(x)

        # Backward scan: right → left (flip, process, flip back)
        x_flip = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_flip)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Gated combination
        gate_input = torch.cat([y_fwd, y_bwd], dim=-1)  # (B, L, 2*d_model)
        g = self.gate(gate_input)  # (B, L, d_model) in [0, 1]
        y = g * y_fwd + (1 - g) * y_bwd

        # Residual connection
        y = self.dropout(y) + residual

        return y


class BiMambaEncoder(nn.Module):
    """
    Stack of L Bidirectional Mamba Blocks forming the temporal encoder.

    Input:  (B, N, d_model) — embedded video features
    Output: (B, N, d_model) — contextual representations
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 n_layers=4, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, N, d_model)

        Returns:
            x: (B, N, d_model)
        """
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return x
