# config.py
"""
VASNet Configuration - Exact paper hyperparameters (Fajtl et al., 2019)
"""
import torch

class Config:
    # ── Paths ────────────────────────────────────────────────
    data_dir     = "./data"
    dataset_name = "tvsum"          # "tvsum" or "summe"

    datasets = {
        "tvsum":   "eccv16_dataset_tvsum_google_pool5.h5",
        "summe":   "eccv16_dataset_summe_google_pool5.h5",
        "ovp":     "eccv16_dataset_ovp_google_pool5.h5",
        "youtube": "eccv16_dataset_youtube_google_pool5.h5",
    }

    # ── Model (EXACT paper values) ───────────────────────────
    input_size   = 1024             # GoogLeNet pool5 dim
    hidden_size  = 1024             # Paper: 1024
    dropout      = 0.5              # Paper: 0.5
    attn_scale   = 0.06             # Paper: Eq.1 scale factor

    # ── Training (EXACT paper values) ────────────────────────
    lr           = 5e-5             # Paper: 5×10⁻⁵
    l2_reg       = 1e-5             # Paper: 10⁻⁵
    epochs       = 300              # Paper: ~200-300
    n_splits     = 5                # Paper: 5-fold cross-validation
    seed         = 12345            # Different seed for variety

    # ── Training Enhancements ────────────────────────────────
    use_scheduler       = True      # Cosine LR decay
    warmup_epochs       = 10
    accumulation_steps  = 1         # Paper uses batch=1
    clip_grad_norm      = 5.0       # Gradient clipping for stability

    # ── GPU settings ─────────────────────────────────────────
    empty_cache_every_n_epochs = 50

    # ── Evaluation ───────────────────────────────────────────
    summary_rate = 0.15             # Paper: max 15% of video length
    eval_every   = 5

    # ── Device (auto-detect) ─────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    # ── Checkpoints ──────────────────────────────────────────
    checkpoint_dir = "./checkpoints"
    results_file   = "./results.txt"