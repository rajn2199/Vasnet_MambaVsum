# Video Summarization Repository

This repository collects three generations of video summarization models:

- **VASNet** as the classic self-attention baseline.
- **FullTransNet** as the Transformer-based attention baseline.
- **MambaVSum** as our proposed model for efficient multimodal summarization.

The main goal of the project is to compare attention-heavy summarization methods with a newer linear-time sequence model. In particular, **MambaVSum is the proposed model** in this repository and is designed to scale in **$O(N)$** time with respect to sequence length, avoiding the quadratic self-attention bottleneck used by older techniques.

## Models

### 1. VASNet

VASNet is the original attention-based summarization network implemented in [model.py](model.py). It follows the paper _Summarizing Videos with Attention_ and uses a single-head multiplicative self-attention block to score frame importance.

Core flow:

```text
Input frame features -> self-attention -> residual + layer norm -> 2-layer MLP -> frame scores
```

Key details:

- Input: GoogLeNet pool5 features with shape `(1, N, 1024)`
- Attention: learnable `U`, `V`, and `C` projections
- Output: per-frame importance scores in `[0, 1]`
- Summary generation: 0/1 knapsack over selected shots

### 2. FullTransNet

FullTransNet is the Transformer-based baseline located under [fulltransnet/](fulltransnet/). It combines local and global attention inside an encoder-decoder architecture.

Core flow:

```text
Video embedding -> local-global encoder -> decoder attention -> score projection
```

Key details:

- Uses Local-Global Multi-Head Attention in the encoder
- Uses standard multi-head attention in the decoder
- Relies on change points / shot boundaries for global context
- Serves as a stronger but more expensive attention baseline

### 3. MambaVSum

MambaVSum is the proposed model in [mambavsum/](mambavsum/). It replaces quadratic attention with a bidirectional state-space encoder and adds multimodal fusion for visual and audio inputs.

Core flow:

```text
Visual + audio features -> multimodal fusion -> BiMamba encoder -> multi-scale pooling -> changepoint attention -> score regressor
```

Key details:

- Supports GoogLeNet, CLIP, and multimodal CLIP + audio inputs
- Uses bidirectional Mamba blocks for temporal modeling
- Includes multi-scale temporal pooling at 1x, 2x, and 4x scales
- Adds sparse change-point attention for shot-level structure
- Produces frame-level importance scores for knapsack-based summary creation
- Designed for linear-time sequence modeling, or **$O(N)$** with respect to video length

## Why MambaVSum

Older video summarization methods usually depend on full self-attention or Transformer-style attention matrices. Those approaches are powerful, but their cost grows quadratically with sequence length, which becomes expensive for long videos.

MambaVSum addresses that limitation by:

- replacing full self-attention with state-space sequence modeling,
- using bidirectional context instead of a single left-to-right scan,
- combining visual and audio evidence,
- and adding multi-scale temporal pooling so the model sees frame-, segment-, and scene-level structure.

That makes MambaVSum the best fit in this repository when the goal is efficient video summarization on longer sequences.

## Repository Layout

```text
Video_Summarization/
├── model.py                 # VASNet baseline
├── train.py                 # VASNet training script
├── evaluate.py              # VASNet evaluation
├── dataset.py               # Dataset loading for ECCV16 features
├── knapsack.py              # Summary selection logic
├── fulltransnet/            # FullTransNet implementation
├── mambavsum/               # Proposed MambaVSum implementation
└── README.md                # Project overview
```

## VASNet Details

VASNet is the cleanest baseline in the repository. It uses learned linear projections to compute a soft attention matrix over all frames, then aggregates context with a residual block and regresses frame importance scores.

Main characteristics:

- single-head soft self-attention
- residual connection with layer normalization
- two-layer MLP regressor
- trained with 5-fold cross validation
- summary extraction with 0/1 knapsack under a fixed summary budget

## FullTransNet Details

FullTransNet expands the baseline into an encoder-decoder Transformer for video summarization.

Main characteristics:

- positional encoding for temporal order
- local-global attention in the encoder
- decoder self-attention and encoder-decoder attention
- change-point aware masking
- heavier than VASNet, but more expressive than a simple single-head attention model

## MambaVSum Details

MambaVSum is the proposed model and the main efficiency-focused contribution of the repository.

Main characteristics:

- multimodal fusion of visual and audio features
- BiMamba encoder for temporal dependency modeling
- multi-scale pooling to capture short- and long-range structure
- change-point sparse attention for boundary-aware refinement
- lightweight score regressor for frame importance prediction

Because the sequence model is linear-time, MambaVSum is intended for longer videos where full attention becomes costly.

## Dataset Setup

The repository uses pre-extracted ECCV16-style HDF5 features for the VASNet and FullTransNet baselines.

Supported datasets:

- TVSum
- SumMe
- OVP
- YouTube

The default configuration uses TVSum.

## Quick Start

### Install Dependencies

```bash
pip install torch numpy h5py
```

For the MambaVSum pipeline, additional packages may be required depending on which feature extractor you use:

- `tqdm`
- `open_clip_torch`
- `opencv-python`
- `torchaudio`

### Train VASNet

```bash
python train.py
```

### Run FullTransNet

Use the scripts inside the [fulltransnet/](fulltransnet/) folder. That subproject has its own training and evaluation entry points.

### Run MambaVSum

Use the scripts inside the [mambavsum/](mambavsum/) folder for the proposed multimodal model.

## Core Baseline Configuration

Edit [config.py](config.py) to control the VASNet baseline:

| Parameter      |   Default | Description                        |
| -------------- | --------: | ---------------------------------- |
| `dataset_name` | `"tvsum"` | Dataset to use                     |
| `input_size`   |    `1024` | GoogLeNet pool5 feature dimension  |
| `hidden_size`  |    `1024` | Hidden dimension for the regressor |
| `dropout`      |     `0.5` | Dropout probability                |
| `attn_scale`   |    `0.06` | Attention scaling factor           |
| `lr`           |    `5e-5` | Learning rate                      |
| `l2_reg`       |    `1e-5` | L2 regularization weight           |
| `epochs`       |     `300` | Maximum training epochs            |
| `n_splits`     |       `5` | Number of cross-validation folds   |
| `summary_rate` |    `0.15` | Maximum summary length             |

## Inference and Summary Generation

After scoring frames, summary extraction uses the 0/1 knapsack formulation to select the best set of shots within the summary budget.

```python
from knapsack import generate_summary

summary = generate_summary(
    pred_scores=scores.cpu().numpy(),
    cps=change_points,
    n_frames=n_frames,
    nfps=n_frame_per_seg,
    picks=picks,
    proportion=0.15,
)
```

## Citation

If you use the baseline VASNet implementation, cite the original paper:

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

For the proposed MambaVSum model, the repository currently uses the project report and code as the primary reference implementation.

## License

This repository is intended for research use. Please check the original papers and dataset licenses before redistribution.
