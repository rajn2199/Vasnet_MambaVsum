# MambaVSum: Efficient Multimodal Video Summarization via Hybrid State Space Models

A novel video summarization architecture combining **Bidirectional Mamba (Selective State Space Models)** with **multimodal fusion** (CLIP visual + audio features) for state-of-the-art performance with O(N) efficiency.

## Key Contributions

1. **Bidirectional Mamba Encoder** — replaces O(N²) self-attention with O(N) selective state space models, enabling efficient processing of long videos
2. **Multimodal Fusion** — first systematic study of CLIP visual + audio features for video summarization on standard benchmarks
3. **Multi-Scale Temporal Pooling** — captures patterns at frame, segment, and scene granularities
4. **Change-Point Sparse Attention** — lightweight global context at shot boundaries

## Architecture

```
Visual Features (GoogLeNet 1024-d / CLIP 768-d)
Audio Features (128-d, optional)
              ↓
  Multimodal Fusion (Gated / CrossAttn)
              ↓
  BiMamba Encoder (L layers, O(N))
    ├── Forward Mamba (left → right)
    ├── Backward Mamba (right → left)
    └── Gated combination
              ↓
  Multi-Scale Temporal Pooling (1×, 2×, 4×)
              ↓
  Changepoint Sparse Attention
              ↓
  Score Regressor → (N,) ∈ [0, 1]
              ↓
  Knapsack Summary Generation
```

## Project Structure

```
mambavsum/
├── config.py                    # All hyperparameters
├── model/
│   ├── __init__.py
│   ├── mamba.py                 # Bidirectional Mamba (pure PyTorch)
│   ├── fusion.py                # Multimodal fusion modules
│   └── mambavsum.py             # Full MambaVSum architecture
├── dataset.py                   # Multi-mode dataset loader
├── train.py                     # Training with 5-fold CV
├── evaluate.py                  # F-score evaluation
├── knapsack.py                  # Summary generation
├── extract_clip_features.py     # CLIP feature extraction
├── extract_audio_features.py    # Audio feature extraction
├── run_ablations.py             # Ablation study runner
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy h5py tqdm
# For CLIP features:
pip install open_clip_torch opencv-python
# For audio features:
pip install torchaudio
```

### 2. Train with GoogLeNet Features (Baseline)

Works immediately with existing ECCV16 data:

```bash
python train.py
```

### 3. Extract CLIP Features (for best results)

**Option A — Real CLIP features** (requires raw videos):
```bash
python extract_clip_features.py --source_h5 ../data/eccv16_dataset_tvsum_google_pool5.h5 --video_dir ./videos/tvsum/ --output_h5 ../data/tvsum_clip_features.h5

**Option B — Synthetic features** (for testing pipeline):
```bash
python extract_clip_features.py --synthetic \
    --source_h5 ../data/eccv16_dataset_tvsum_google_pool5.h5 \
    --output_h5 ../data/tvsum_clip_features.h5
```

Then update `config.py`:
```python
feature_mode = "clip"  # or "multimodal"
```

### 4. Run Ablation Study

```bash
python run_ablations.py
```

## Feature Modes

| Mode | Input | Dimension | Description |
|---|---|---|---|
| `googlenet` | GoogLeNet pool5 | 1024 | Original ECCV16 features |
| `clip` | CLIP ViT-L/14 | 768 | Modern visual features |
| `multimodal` | CLIP + Audio | 768 + 128 | Full multimodal fusion |

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `mamba_d_model` | 256 | Internal model dimension |
| `mamba_d_state` | 16 | SSM state expansion |
| `mamba_n_layers` | 4 | Number of BiMamba layers |
| `mamba_expand` | 2 | Block expansion factor |
| `temporal_scales` | [1, 2, 4] | Multi-scale pooling strides |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 200 | Max training epochs |

## Citation

```bibtex
@article{mambavsum2026,
  title={MambaVSum: Efficient Multimodal Video Summarization via Hybrid State Space Models},
  author={Your Name},
  year={2026}
}
```
