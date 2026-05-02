# run_ablations.py
"""
MambaVSum Ablation Study Runner.

Runs multiple configurations to build Table 3/4 of the paper:
  1. GoogLeNet + VASNet baseline (attention)
  2. GoogLeNet + MambaVSum (architecture improvement)
  3. CLIP + MambaVSum (feature improvement)
  4. CLIP + Audio + MambaVSum (multimodal improvement)

Each configuration is trained with 5-fold CV and results are logged.

Usage:
    python run_ablations.py
"""
import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from config import Config
from dataset import get_keys, make_splits, VideoDataset
from model.mambavsum import MambaVSum
from evaluate import evaluate_dataset
from train import train_one_split


ABLATION_CONFIGS = [
    # (name, feature_mode, mamba_layers, mamba_d_model, scales)
    {
        "name": "GoogLeNet + MambaVSum-Tiny",
        "feature_mode": "googlenet",
        "mamba_n_layers": 2,
        "mamba_d_model": 128,
        "temporal_scales": [1, 2],
    },
    {
        "name": "GoogLeNet + MambaVSum-Base",
        "feature_mode": "googlenet",
        "mamba_n_layers": 4,
        "mamba_d_model": 256,
        "temporal_scales": [1, 2, 4],
    },
    {
        "name": "GoogLeNet + MambaVSum-Large",
        "feature_mode": "googlenet",
        "mamba_n_layers": 6,
        "mamba_d_model": 384,
        "temporal_scales": [1, 2, 4],
    },
    {
        "name": "CLIP + MambaVSum-Base",
        "feature_mode": "clip",
        "mamba_n_layers": 4,
        "mamba_d_model": 256,
        "temporal_scales": [1, 2, 4],
    },
    {
        "name": "CLIP + Audio + MambaVSum-Base",
        "feature_mode": "multimodal",
        "mamba_n_layers": 4,
        "mamba_d_model": 256,
        "temporal_scales": [1, 2, 4],
    },
]


def run_ablation(ablation_cfg, dataset_name="tvsum"):
    """Run one ablation configuration."""
    cfg = Config()
    cfg.dataset_name = dataset_name

    # Apply ablation settings
    cfg.feature_mode = ablation_cfg["feature_mode"]
    cfg.mamba_n_layers = ablation_cfg["mamba_n_layers"]
    cfg.mamba_d_model = ablation_cfg["mamba_d_model"]
    cfg.temporal_scales = ablation_cfg["temporal_scales"]

    # Shorter training for ablations
    cfg.epochs = 150
    cfg.patience = 40

    h5_path = os.path.join(cfg.data_dir, cfg.datasets[cfg.dataset_name])
    if not os.path.exists(h5_path):
        print(f"  [ERROR] Dataset not found: {h5_path}")
        return None

    keys = get_keys(h5_path)
    splits = make_splits(keys, cfg.n_splits, cfg.seed)

    all_f = []
    for i, split in enumerate(splits):
        print(f"  Split {i+1}/{cfg.n_splits}...")
        if cfg.use_amp:
            torch.cuda.empty_cache()
        f = train_one_split(i, split, h5_path, cfg)
        all_f.append(f)

    mean_f = float(np.mean(all_f))
    std_f = float(np.std(all_f))

    # Count params
    model = MambaVSum(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "name": ablation_cfg["name"],
        "mean_f": mean_f,
        "std_f": std_f,
        "per_split": all_f,
        "n_params": n_params,
    }


def main():
    print("=" * 70)
    print("  MambaVSum ABLATION STUDY")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)

    results = []

    for i, ablation_cfg in enumerate(ABLATION_CONFIGS):
        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/{len(ABLATION_CONFIGS)}] {ablation_cfg['name']}")
        print(f"{'=' * 70}")

        start = time.time()
        result = run_ablation(ablation_cfg)
        elapsed = time.time() - start

        if result:
            result["time_seconds"] = elapsed
            results.append(result)

            print(f"\n  [OK] {result['name']}: "
                  f"{result['mean_f']:.2f}% ± {result['std_f']:.2f}% "
                  f"({result['n_params']:,} params, {elapsed:.0f}s)")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<40} {'F-score':>10} {'Params':>12}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 12}")

    for r in results:
        print(f"  {r['name']:<40} "
              f"{r['mean_f']:.2f}±{r['std_f']:.1f}% "
              f"{r['n_params']:>10,}")

    # Save results
    results_path = "./ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
