# train.py
"""
MambaVSum Training — Full training loop with 5-fold cross-validation.

Supports:
  - GoogLeNet features (backward compat with ECCV16)
  - CLIP features (modern visual backbone)
  - Multimodal (CLIP + audio) features
  - Mixed precision training (AMP)
  - Cosine LR scheduling with warmup
  - Early stopping
  - Gradient clipping
"""
import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import Config
from model.mambavsum import MambaVSum
from dataset import VideoDataset, get_keys, make_splits
from evaluate import evaluate_dataset


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                     min_lr_ratio=0.01):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_split(split_id, split, h5_path, cfg):
    """Train and evaluate on one fold."""

    model = MambaVSum(cfg).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.use_scheduler:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, cfg.warmup_epochs, cfg.epochs
        )

    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)

    # Determine CLIP h5 path
    clip_h5 = None
    if cfg.feature_mode in ("clip", "multimodal"):
        clip_h5 = os.path.join(cfg.data_dir, cfg.clip_datasets[cfg.dataset_name])
        if not os.path.exists(clip_h5):
            print(f"  [WARN] CLIP features not found: {clip_h5}")
            print(f"    Run: python extract_clip_features.py --synthetic")
            print(f"    Falling back to GoogLeNet features.")
            cfg.feature_mode = "googlenet"
            clip_h5 = None

    train_ds = VideoDataset(h5_path, split["train"], cfg.feature_mode, clip_h5)
    test_ds = VideoDataset(h5_path, split["test"], cfg.feature_mode, clip_h5)

    best_f = 0.0
    best_epoch = 0
    patience_counter = 0

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + split_id)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        order = rng.permutation(len(train_ds))
        epoch_start = time.time()

        for sample_idx, idx in enumerate(order):
            sample = train_ds[int(idx)]
            feat = sample["features"].unsqueeze(0).to(cfg.device)
            gt = sample["gtscore"].to(cfg.device)

            # Prepare audio features
            audio = None
            if cfg.feature_mode == "multimodal" and "audio_features" in sample:
                audio = sample["audio_features"].unsqueeze(0).to(cfg.device)

            # Change points for sparse attention
            change_points = sample.get("change_points", None)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                pred, _ = model(feat, audio, change_points)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            max_norm=cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Progress indicator every 10 samples
            if (sample_idx + 1) % 10 == 0:
                elapsed = time.time() - epoch_start
                print(f"    Ep {epoch} [{sample_idx+1}/{len(order)}] "
                      f"loss={loss.item():.5f}  {elapsed:.0f}s", flush=True)

        avg_loss = total_loss / len(train_ds)
        epoch_time = time.time() - epoch_start
        print(f"  Ep {epoch:3d}/{cfg.epochs}  "
              f"loss={avg_loss:.5f}  time={epoch_time:.1f}s", flush=True)

        if scheduler is not None:
            scheduler.step()

        # Evaluate
        if epoch % cfg.eval_every == 0:
            fscore = evaluate_dataset(model, test_ds, cfg)

            if fscore > best_f:
                best_f, best_epoch = fscore, epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "fscore": fscore,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "feature_mode": cfg.feature_mode,
                            "mamba_d_model": cfg.mamba_d_model,
                            "mamba_n_layers": cfg.mamba_n_layers,
                        },
                    },
                    os.path.join(cfg.checkpoint_dir, f"best_split{split_id}.pt"),
                )
            else:
                patience_counter += cfg.eval_every

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Split {split_id+1}  "
                f"Ep {epoch:3d}/{cfg.epochs}  "
                f"loss={avg_loss:.5f}  "
                f"lr={lr:.1e}  "
                f"F={fscore:.2f}%  "
                f"best={best_f:.2f}% (ep {best_epoch})"
            )

            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if cfg.use_amp and epoch % cfg.empty_cache_every_n_epochs == 0:
            torch.cuda.empty_cache()

    return best_f


def main():
    cfg = Config()

    print("=" * 60)
    print(f"  MambaVSum — {cfg.dataset_name.upper()}")
    print(f"  Feature Mode: {cfg.feature_mode}")
    print(f"  Device: {cfg.device}")
    print(f"  PyTorch: {torch.__version__}")
    if cfg.use_amp:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU: {name} ({vram/1e9:.1f} GB)")
    print(f"  Mamba: d_model={cfg.mamba_d_model}, layers={cfg.mamba_n_layers}, "
          f"d_state={cfg.mamba_d_state}, expand={cfg.mamba_expand}")
    print(f"  Training: lr={cfg.lr}, wd={cfg.weight_decay}, epochs={cfg.epochs}")
    print("=" * 60)

    h5_path = os.path.join(cfg.data_dir, cfg.datasets[cfg.dataset_name])

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"\n  [ERROR] Dataset not found: {h5_path}"
            "\n  Download from: "
            "https://zenodo.org/record/4884870/files/datasets.tar"
        )

    keys = get_keys(h5_path)
    splits = make_splits(keys, cfg.n_splits, cfg.seed)
    print(f"  Videos: {len(keys)}  |  Splits: {cfg.n_splits}")
    print("=" * 60)

    all_f = []

    for i, split in enumerate(splits):
        print(f"\n{'-' * 60}")
        print(f"  SPLIT {i+1}/{cfg.n_splits}  "
              f"({len(split['train'])} train / {len(split['test'])} test)")
        print(f"{'-' * 60}")

        if cfg.use_amp:
            torch.cuda.empty_cache()

        f = train_one_split(i, split, h5_path, cfg)
        all_f.append(f)

        print(f"\n  [OK] Split {i+1} best F-score: {f:.2f}%")

    mean_f = float(np.mean(all_f))
    std_f = float(np.std(all_f))

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS — {cfg.dataset_name.upper()} [{cfg.feature_mode}]")
    print(f"  Per-split : {[f'{v:.2f}%' for v in all_f]}")
    print(f"  Mean F    : {mean_f:.2f}%  ±  {std_f:.2f}%")
    print(f"{'=' * 60}")

    with open(cfg.results_file, "w") as f:
        f.write(f"Model    : MambaVSum\n")
        f.write(f"Dataset  : {cfg.dataset_name}\n")
        f.write(f"Features : {cfg.feature_mode}\n")
        f.write(f"Config   : d_model={cfg.mamba_d_model}, "
                f"layers={cfg.mamba_n_layers}\n")
        f.write(f"Per-split: {all_f}\n")
        f.write(f"Mean F   : {mean_f:.2f}% ± {std_f:.2f}%\n")


if __name__ == "__main__":
    main()
