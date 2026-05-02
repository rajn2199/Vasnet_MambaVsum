# dataset.py
"""
MambaVSum Dataset — Supports GoogLeNet, CLIP, and Multimodal features.
Compatible with ECCV16 standard h5 format.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Loads pre-extracted features from .h5 file(s).

    Supports three modes:
      - "googlenet": original 1024-d features from ECCV16 h5
      - "clip": 768-d CLIP features from separate h5
      - "multimodal": CLIP visual + audio features

    Each item is ONE full video (variable length N frames).
    """
    def __init__(self, h5_path, keys, feature_mode="googlenet",
                 clip_h5_path=None):
        self.h5_path = h5_path
        self.keys = keys
        self.feature_mode = feature_mode
        self.clip_h5_path = clip_h5_path or h5_path

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # Load base data (metadata always from original h5)
        with h5py.File(self.h5_path, "r") as f:
            video = f[key]
            gtscore = torch.from_numpy(video["gtscore"][()]).float()
            gtsummary = np.array(video["gtsummary"][()])
            change_points = np.array(video["change_points"][()])
            n_frames = int(video["n_frames"][()])
            n_frame_per_seg = np.array(video["n_frame_per_seg"][()])
            picks = np.array(video["picks"][()])
            user_summary = np.array(video["user_summary"][()])

            if self.feature_mode == "googlenet":
                features = torch.from_numpy(video["features"][()]).float()

        # Load CLIP / multimodal features
        if self.feature_mode in ("clip", "multimodal"):
            with h5py.File(self.clip_h5_path, "r") as f:
                features = torch.from_numpy(f[key]["features"][()]).float()

                audio_features = None
                if self.feature_mode == "multimodal":
                    if "audio_features" in f[key]:
                        audio_features = torch.from_numpy(
                            f[key]["audio_features"][()]
                        ).float()
                    else:
                        # Fallback: zeros
                        audio_features = torch.zeros(features.shape[0], 128)

        # Normalize ground truth scores to [0, 1]
        gtscore = gtscore - gtscore.min()
        if gtscore.max() > 1e-8:
            gtscore = gtscore / gtscore.max()

        sample = {
            "features": features,
            "gtscore": gtscore,
            "gtsummary": gtsummary,
            "change_points": change_points,
            "n_frames": n_frames,
            "n_frame_per_seg": n_frame_per_seg,
            "picks": picks,
            "user_summary": user_summary,
            "key": key,
        }

        if self.feature_mode == "multimodal" and audio_features is not None:
            sample["audio_features"] = audio_features

        return sample


def get_keys(h5_path):
    """Return all video keys in the h5 file."""
    with h5py.File(h5_path, "r") as f:
        return sorted(list(f.keys()))


def make_splits(keys, n_splits=5, seed=42):
    """Generate n_splits random 80/20 train/test splits."""
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
