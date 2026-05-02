# train.py
"""
VASNet Training - Following paper protocol exactly
"""
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math

from config import Config
from model import VASNet
from dataset import VideoDataset, get_keys, make_splits
from evaluate import evaluate_dataset


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_split(
    split_id: int,
    split: dict,
    h5_path: str,
    cfg: Config,
) -> float:

    model = VASNet(cfg).to(cfg.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    
    # Paper uses Adam with the specified hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.l2_reg,
    )
    
    # Learning rate scheduler
    scheduler = None
    if getattr(cfg, 'use_scheduler', False):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            getattr(cfg, 'warmup_epochs', 10),
            cfg.epochs
        )
    
    criterion = nn.MSELoss()

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)

    train_ds = VideoDataset(h5_path, split["train"])
    test_ds  = VideoDataset(h5_path, split["test"])

    best_f     = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 60  # Early stopping patience
    
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed + split_id)
    
    eval_every = getattr(cfg, 'eval_every', 10)
    clip_grad = getattr(cfg, 'clip_grad_norm', 5.0)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        order = rng.permutation(len(train_ds))

        for idx in order:
            sample = train_ds[int(idx)]
            feat   = sample["features"].unsqueeze(0).to(cfg.device)
            gt     = sample["gtscore"].to(cfg.device)

            optimizer.zero_grad(set_to_none=True)

            # Forward with AMP
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                pred, _ = model(feat)
                loss = criterion(pred, gt)

            # Backward with scaler
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_ds)
        
        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        # Evaluate every N epochs
        if epoch % eval_every == 0:
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
                        "scaler_state_dict": scaler.state_dict(),
                        "config": {
                            "input_size":  cfg.input_size,
                            "hidden_size": cfg.hidden_size,
                            "dropout":     cfg.dropout,
                        },
                    },
                    os.path.join(cfg.checkpoint_dir, f"best_split{split_id}.pt"),
                )
            else:
                patience_counter += eval_every

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Split {split_id+1}  "
                f"Ep {epoch:3d}/{cfg.epochs}  "
                f"loss={avg_loss:.5f}  "
                f"lr={current_lr:.1e}  "
                f"F={fscore:.2f}%  "
                f"best={best_f:.2f}% (ep {best_epoch})"
            )
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Flush GPU cache periodically
        if cfg.use_amp and epoch % cfg.empty_cache_every_n_epochs == 0:
            torch.cuda.empty_cache()

    return best_f


def main():
    cfg = Config()

    print("=" * 60)
    print(f"  VASNet — {cfg.dataset_name.upper()} (Paper Configuration)")
    print(f"  Device : {cfg.device}")
    print(f"  PyTorch: {torch.__version__}")
    if cfg.use_amp:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU    : {name}  ({vram/1e9:.1f} GB VRAM)")
        print(f"  AMP    : enabled")
    print(f"  Config : hidden={cfg.hidden_size}, dropout={cfg.dropout}, "
          f"scale={cfg.attn_scale}")
    print(f"  Training: lr={cfg.lr}, l2={cfg.l2_reg}, epochs={cfg.epochs}")
    print("=" * 60)

    h5_path = os.path.join(cfg.data_dir, cfg.datasets[cfg.dataset_name])

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"\n  ✗ Dataset not found: {h5_path}"
            "\n  Download from: "
            "https://zenodo.org/record/4884870/files/datasets.tar"
        )

    keys = get_keys(h5_path)
    splits = make_splits(keys, cfg.n_splits, cfg.seed)
    print(f"  Videos : {len(keys)}  |  Splits: {cfg.n_splits}")
    print("=" * 60)

    all_f: list[float] = []

    for i, split in enumerate(splits):
        print(f"\n{'─'*60}")
        print(f"  SPLIT {i+1}/{cfg.n_splits}  "
              f"({len(split['train'])} train / {len(split['test'])} test)")
        print(f"{'─'*60}")

        if cfg.use_amp:
            torch.cuda.empty_cache()

        f = train_one_split(i, split, h5_path, cfg)
        all_f.append(f)

        print(f"\n  ✓ Split {i+1} best F-score: {f:.2f}%")

    # Final summary
    mean_f = float(np.mean(all_f))
    std_f  = float(np.std(all_f))

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {cfg.dataset_name.upper()}")
    print(f"  Per-split : {[f'{v:.2f}%' for v in all_f]}")
    print(f"  Mean F    : {mean_f:.2f}%  ±  {std_f:.2f}%")
    print(f"  Target    : TvSum ≈ 61.4%  |  SumMe ≈ 49.7%")
    print(f"{'='*60}")

    # Save results
    with open(cfg.results_file, "w") as f:
        f.write(f"Dataset : {cfg.dataset_name}\n")
        f.write(f"Config  : hidden={cfg.hidden_size}, dropout={cfg.dropout}\n")
        f.write(f"Per-split: {all_f}\n")
        f.write(f"Mean F  : {mean_f:.2f}% ± {std_f:.2f}%\n")
        f.write(f"Target  : TVSum 61.4% / SumMe 49.7%\n")


if __name__ == "__main__":
    main()