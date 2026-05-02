# dataset.py
"""
VASNet Dataset - Loading pre-extracted features from h5 files.
Compatible with the standard ECCV16 dataset format.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Loads pre-extracted GoogLeNet pool5 features from .h5 file.
    
    Each item is ONE full video (variable length N frames).
    VASNet processes one video at a time (batch size = 1).
    
    Dataset format (ECCV16 standard):
    - features: (N, 1024) GoogLeNet pool5 features
    - gtscore: (N,) frame-level importance scores [0, 1]
    - change_points: (K, 2) segment boundaries [start, end]
    - n_frames: total original frame count
    - n_frame_per_seg: (K,) frames per segment
    - picks: (N,) subsampled frame indices
    - user_summary: (n_users, n_frames) binary user annotations
    """

    def __init__(self, h5_path: str, keys: list[str]):
        self.h5_path = h5_path
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx]
        
        with h5py.File(self.h5_path, "r") as f:
            video = f[key]
            
            # Features: (N, 1024) - subsampled frames
            features = torch.from_numpy(video["features"][()]).float()
            
            # Ground truth importance scores: (N,)
            gtscore = torch.from_numpy(video["gtscore"][()]).float()
            
            # Ground truth binary summary: (N,) - for reference
            gtsummary = np.array(video["gtsummary"][()])
            
            # Temporal segmentation info
            change_points = np.array(video["change_points"][()])  # (K, 2)
            n_frames = int(video["n_frames"][()])                  # total original frames
            n_frame_per_seg = np.array(video["n_frame_per_seg"][()])  # (K,)
            
            # Subsampling indices: maps subsampled -> original frame index
            picks = np.array(video["picks"][()])  # (N,)
            
            # User annotations: (n_users, n_frames) binary
            user_summary = np.array(video["user_summary"][()])
        
        # Normalize ground truth scores to [0, 1]
        # This is important for MSE loss to work properly
        gtscore = gtscore - gtscore.min()
        if gtscore.max() > 1e-8:
            gtscore = gtscore / gtscore.max()

        return {
            "features": features,              # (N, 1024) float32
            "gtscore": gtscore,                # (N,) float32 normalized
            "gtsummary": gtsummary,            # (N,) int binary
            "change_points": change_points,    # (K, 2) int
            "n_frames": n_frames,              # scalar int
            "n_frame_per_seg": n_frame_per_seg,  # (K,) int
            "picks": picks,                    # (N,) int
            "user_summary": user_summary,      # (n_users, n_frames) binary
            "key": key,                        # video identifier
        }


def get_keys(h5_path: str) -> list[str]:
    """Return all video keys in the h5 file."""
    with h5py.File(h5_path, "r") as f:
        return sorted(list(f.keys()))  # Sort for reproducibility


def make_splits(
    keys: list[str], 
    n_splits: int = 5, 
    seed: int = 42
) -> list[dict]:
    """
    Generate n_splits random 80/20 train/test splits.
    
    Paper protocol: 5-fold random splits, 80% train / 20% test
    Results are averaged across all splits.
    """
    rng = np.random.default_rng(seed)
    n = len(keys)
    splits = []

    for _ in range(n_splits):
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        splits.append({
            "train": [keys[i] for i in perm[:cut]],
            "test": [keys[i] for i in perm[cut:]],
        })

    return splits