# evaluate.py
"""
MambaVSum Evaluation — F-score computation following ECCV16 protocol.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from knapsack import generate_summary
from config import Config


def evaluate_summary(machine_summary, user_summary):
    """Compute F-score between machine and one user summary."""
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)

    overlap = np.sum(machine_summary * user_summary)
    precision = overlap / (np.sum(machine_summary) + 1e-8)
    recall = overlap / (np.sum(user_summary) + 1e-8)

    if precision + recall > 0:
        return float(2 * precision * recall / (precision + recall))
    return 0.0


@torch.no_grad()
def evaluate_dataset(model, dataset, cfg):
    """
    Evaluate model on dataset using standard video summarization protocol.
    Returns: mean F-score (percentage)
    """
    model.eval()
    video_fscores = []

    for sample in dataset:
        feat = sample["features"].unsqueeze(0).to(cfg.device)

        # Prepare audio if multimodal
        audio = None
        if cfg.feature_mode == "multimodal" and "audio_features" in sample:
            audio = sample["audio_features"].unsqueeze(0).to(cfg.device)

        # Change points for sparse attention
        change_points = sample.get("change_points", None)

        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            pred_scores, _ = model(feat, audio, change_points)

        pred_scores = pred_scores.cpu().numpy()

        # Normalize to [0, 1]
        score_min, score_max = pred_scores.min(), pred_scores.max()
        if score_max - score_min > 1e-8:
            pred_scores = (pred_scores - score_min) / (score_max - score_min)
        else:
            pred_scores = np.zeros_like(pred_scores)

        # Generate summary via knapsack
        machine_summary = generate_summary(
            pred_scores,
            sample["change_points"],
            int(sample["n_frames"]),
            sample["n_frame_per_seg"],
            sample["picks"],
            cfg.summary_rate,
        )

        # Compare against each user
        user_summary = sample["user_summary"]
        n_frames = user_summary.shape[1]

        if len(machine_summary) > n_frames:
            machine_summary = machine_summary[:n_frames]
        elif len(machine_summary) < n_frames:
            machine_summary = np.concatenate([
                machine_summary,
                np.zeros(n_frames - len(machine_summary), dtype=np.float32)
            ])

        user_fscores = []
        for u in range(user_summary.shape[0]):
            f = evaluate_summary(machine_summary, user_summary[u])
            user_fscores.append(f)

        video_fscores.append(float(np.mean(user_fscores)))

    return float(np.mean(video_fscores)) * 100.0
