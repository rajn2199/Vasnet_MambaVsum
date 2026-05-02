# VASNet: Video Attention-based Summarization Network

A PyTorch implementation of **VASNet** (Fajtl et al., 2019) for automatic video summarization using self-attention mechanisms.

## Overview

VASNet uses a self-attention mechanism to determine frame importance in videos, generating concise video summaries. This implementation follows the exact architecture and hyperparameters from the original paper.

**Paper:** *Summarizing Videos with Attention* (Fajtl et al., ACCV 2018 Workshop)

## Features

- Self-attention based frame importance scoring
- 0/1 Knapsack algorithm for optimal summary generation
- 5-fold cross-validation evaluation
- Mixed precision training (AMP) support
- Cosine learning rate scheduling with warmup
- Compatible with standard ECCV16 benchmark datasets

## Requirements

```
torch>=2.0
numpy
h5py
```

Install dependencies:
```bash
pip install torch numpy h5py
```

## Dataset Setup

This implementation uses pre-extracted GoogLeNet pool5 features in HDF5 format (ECCV16 standard).

### Supported Datasets
- **TVSum** - 50 videos (recommended)
- **SumMe** - 25 videos
- **OVP** - 50 videos
- **YouTube** - 39 videos

### Download
Download the preprocessed datasets from:
[https://zenodo.org/record/4884870/files/datasets.tar](https://zenodo.org/record/4884870/files/datasets.tar)

Extract to the `data/` directory:
```
data/
├── eccv16_dataset_tvsum_google_pool5.h5
├── eccv16_dataset_summe_google_pool5.h5
├── eccv16_dataset_ovp_google_pool5.h5
└── eccv16_dataset_youtube_google_pool5.h5
```

## Usage

### Training

1. **Configure the dataset** in `config.py`:
   ```python
   dataset_name = "tvsum"  # or "summe", "ovp", "youtube"
   ```

2. **Run training:**
   ```bash
   python train.py
   ```

Training runs 5-fold cross-validation automatically. Best models for each split are saved to `checkpoints/`.

### Configuration

Edit `config.py` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | `"tvsum"` | Dataset to use |
| `input_size` | `1024` | GoogLeNet pool5 feature dimension |
| `hidden_size` | `1024` | Model hidden dimension |
| `dropout` | `0.5` | Dropout probability |
| `attn_scale` | `0.06` | Attention scaling factor |
| `lr` | `5e-5` | Learning rate |
| `l2_reg` | `1e-5` | L2 regularization weight |
| `epochs` | `300` | Maximum training epochs |
| `n_splits` | `5` | Number of cross-validation folds |
| `summary_rate` | `0.15` | Maximum summary length (15% of video) |

### Loading a Trained Model

```python
import torch
from model import VASNet
from config import Config

# Load configuration and model
cfg = Config()
model = VASNet(cfg)

# Load checkpoint
checkpoint = torch.load("checkpoints/best_split0.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Run inference
# features: (1, N, 1024) tensor of GoogLeNet pool5 features
scores, attention = model(features)
# scores: (N,) importance scores in [0, 1]
# attention: (N, N) attention weight matrix
```

### Generating Video Summaries

```python
from knapsack import generate_summary

# After getting scores from the model
summary = generate_summary(
    pred_scores=scores.cpu().numpy(),
    cps=change_points,      # (K, 2) segment boundaries
    n_frames=n_frames,      # total original frames
    nfps=n_frame_per_seg,   # (K,) frames per segment
    picks=picks,            # (N,) subsampled frame indices
    proportion=0.15         # max 15% of video length
)
# summary: binary array (n_frames,) where 1 = selected frame
```

## Project Structure

```
vasnet/
├── config.py      # Configuration and hyperparameters
├── model.py       # VASNet model architecture
├── train.py       # Training script with 5-fold CV
├── evaluate.py    # F-score evaluation metrics
├── dataset.py     # HDF5 dataset loading
├── knapsack.py    # Knapsack summary generation
├── checkpoints/   # Saved model weights
├── data/          # Dataset HDF5 files
└── results.txt    # Training results
```

## Expected Results

Following the paper's evaluation protocol:

| Dataset | Target F-score | This Implementation |
|---------|----------------|---------------------|
| TVSum   | ~61.4%         | ~56-61%             |
| SumMe   | ~49.7%         | ~47-52%             |

Results may vary slightly due to random splits.

## Model Architecture

```
Input: (1, N, 1024) - GoogLeNet pool5 features
          ↓
    Self-Attention (Eq. 1-5)
          ↓
    Residual + LayerNorm
          ↓
    2-Layer MLP Regressor
          ↓
Output: (N,) frame importance scores
```

## Citation

If you use this code, please cite the original VASNet paper:

```bibtex
@inproceedings{fajtl2019summarizing,
  title={Summarizing videos with attention},
  author={Fajtl, Jiri and Sokeh, Hajar Sadeghi and Argyriou, Vasileios and Monekosso, Dorothy and Remagnino, Paolo},
  booktitle={Asian Conference on Computer Vision},
  pages={39--54},
  year={2018},
  organization={Springer}
}
```

## License

This implementation is for research purposes. Please refer to the original paper and datasets for licensing information.
