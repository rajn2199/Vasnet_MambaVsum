# FullTransNet: Full Transformer with Local-Global Attention for Video Summarization

A pure PyTorch implementation of the FullTransNet paper (Lan et al., 2024) for video summarization.

## Key Features

- **Local-Global Attention**: Sliding-window local attention + global attention on change-point frames
- **Encoder-Decoder Transformer**: 6-layer encoder with LGA + 6-layer decoder with standard MHA
- **Pure PyTorch**: No dependency on Longformer or custom CUDA kernels — runs on any platform
- **TVSum Dataset**: Configured to use the TVSum dataset from the parent `data/` folder

## Project Structure

```
fulltransnet/
├── train.py              # Main training entry point
├── evaluate.py           # Evaluation script
├── make_split.py         # Generate train/test splits
├── make_shots.py         # Compute shot boundaries (KTS)
├── requirements.txt      # Python dependencies
├── model/
│   ├── attention.py      # Local-Global Attention (pure PyTorch)
│   ├── transformer.py    # Encoder-Decoder Transformer
│   ├── losses.py         # Loss functions
│   └── train_loop.py     # Training loop per split
├── helpers/
│   ├── data_helper.py    # Dataset loading & utilities
│   ├── vsumm_helper.py   # Video summary evaluation (F1, knapsack)
│   └── init_helper.py    # Argument parsing & initialization
├── kts/
│   ├── cpd_auto.py       # Auto change-point detection
│   └── cpd_nonlin.py     # Non-linear change-point detection
└── splits/
    └── tvsum.yml          # 5-fold cross-validation splits
```

## Setup

```bash
pip install -r requirements.txt
```

The TVSum dataset (`eccv16_dataset_tvsum_google_pool5.h5`) should be in `../data/` relative to this folder.

## Training

Train on TVSum with default settings (300 epochs, BCE loss):

```bash
python train.py
```

Custom training:

```bash
python train.py --max-epoch 100 --lr 0.0013 --loss bce --device cuda
python train.py --device cpu --max-epoch 10   # Quick test on CPU
```

## Evaluation

Evaluate saved checkpoints:

```bash
python evaluate.py --model-dir ./model_save/tvsum --splits ./splits/tvsum.yml
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Input dim | 1024 (GoogLeNet pool5) |
| Hidden dim | 64 |
| Attention heads | 8 |
| Encoder layers | 6 (Local-Global Attention) |
| Decoder layers | 6 (Standard Multi-Head Attention) |
| Window size | 16 |
| FFN dim | 2048 |
| Max sequence length | 1536 |

## Detailed Performance Comparison

| Metric | FullTransNet | VASNet | MambaVSum |
|--------|------------|--------|-----------|
| **Feature Modality** | Unimodal (GoogLeNet) | Unimodal (GoogLeNet) | Multimodal CLIP+ Audio |
| **Core Architecture** | Encoder-Decoder Transformer with Local-Global Attention | Soft Self-Attention | BiMamba + Multi-Scale Pooling |
| **Complexity** | O(N) | O(N²) | O(N) |
| **Parameters** | ~3.7 Million | ~2.1 Million | 756,865 (3x smaller) |
| **Training Time** | ~50 min (300 epochs) | ~15 min | ~10 min (~4s per epoch) |
| **Split 1 F-Score** | 52.00% | 57.89% | 56.38% |
| **Split 2 F-Score** | 52.88% | 55.02% | 57.79% |
| **Split 3 F-Score** | 59.45% | 51.13% | 52.82% |
| **Split 4 F-Score** | 49.35% | 57.88% | 55.12% |
| **Split 5 F-Score** | 58.80% | 60.08% | 59.22% |
| **Mean F-Score ± Std** | **54.50% ± 5.00%** | 56.40% ± 3.09% | 56.27% ± 2.20% (more stable) |

### Key Observations

- **FullTransNet** offers a pure PyTorch implementation with linear-time complexity via Local-Global Attention, making it efficient and portable across platforms
- **Performance Trade-off**: Slightly lower mean score than VASNet/MambaVSum but competitive, with room for hyperparameter tuning and longer training
- **Stability**: Higher variance (±5.00%) compared to MambaVSum (±2.20%), suggesting sensitivity to split distribution
- **Efficiency**: ~50% more parameters than VASNet but achieves O(N) complexity for better scalability on long sequences
- **No External Dependencies**: Unlike Longformer-based approaches, runs on standard PyTorch without custom CUDA kernels

## Acknowledgments

Based on the [FullTransNet](https://github.com/ChiangLu/FullTransNet) paper and official implementation.
Uses datasets from [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) and pipeline from [VASNet](https://github.com/ok1zjf/VASNet).
