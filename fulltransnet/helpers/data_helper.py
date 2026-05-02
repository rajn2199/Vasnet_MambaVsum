# helpers/data_helper.py
"""
Dataset loading and utility functions for FullTransNet.
Handles HDF5 video datasets and provides data loading, YAML I/O,
and checkpoint management.
"""
import random
from os import PathLike
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import yaml


class VideoDataset(object):
    """Dataset class that loads video features from HDF5 files.

    Each item returns:
      key, seq, seqdiff, gtscore, cps, n_frames, nfps, picks, user_summary, gt_summary
    """

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_name = video_path.name
        video_file = self.datasets[dataset_name][video_name]

        seq = video_file['features'][...].astype(np.float32)
        seq_size = seq.shape

        # Compute frame differences
        seqdiff = []
        seq_temp = 0
        for j in range(seq_size[0]):
            if j == 0:
                seq_current = seq[j]
            else:
                seq_current = seq[j]
                seq_temp = seq_current - seq_before
                seqdiff.append(seq_temp)
            seq_before = seq_current
        seqdiff.append(seq_temp)
        seqdiff = np.array(seqdiff)

        gtscore = video_file['gtscore'][...].astype(np.float32)
        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)

        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        gt_summary = video_file['gtsummary'][...].astype(np.float32)

        # Normalize gtscore to [0, 1]
        gtscore -= gtscore.min()
        if gtscore.max() > 0:
            gtscore /= gtscore.max()

        return key, seq, seqdiff, gtscore, cps, n_frames, nfps, picks, user_summary, gt_summary

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]):
        """Open all unique HDF5 files referenced by the keys."""
        dataset_paths = {str(Path(key).parent) for key in keys}
        datasets = {path: h5py.File(path, 'r') for path in dataset_paths}
        return datasets


class DataLoader(object):
    """Simple data loader that iterates one video at a time (batch_size=1)."""

    def __init__(self, dataset: VideoDataset, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.data_idx = list(range(len(self.dataset)))

    def __iter__(self):
        self.iter_idx = 0
        if self.shuffle:
            random.shuffle(self.data_idx)
        return self

    def __next__(self):
        if self.iter_idx == len(self.dataset):
            raise StopIteration
        curr_idx = self.data_idx[self.iter_idx]
        batch = self.dataset[curr_idx]
        self.iter_idx += 1
        return batch


class AverageMeter(object):
    """Tracks running averages for named metrics."""

    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str):
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str):
        assert attr in self.totals and attr in self.counts


def get_ckpt_dir(model_dir: PathLike):
    """Return checkpoint directory path."""
    return Path(model_dir) / 'checkpoint'


def get_ckpt_path(model_dir: PathLike, split_path: PathLike, split_index: int):
    """Return checkpoint file path for a specific split."""
    split_path = Path(split_path)
    return get_ckpt_dir(model_dir) / f'{split_path.name}.{split_index}.pt'


def load_yaml(path: PathLike):
    """Load a YAML file."""
    with open(path) as f:
        obj = yaml.safe_load(f)
    return obj


def dump_yaml(obj: Any, path: PathLike):
    """Dump object to a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(obj, f)
