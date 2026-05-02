# evaluate.py
"""
VASNet Evaluation - Exact protocol from the paper.
F-score computation following Zhang et al. (ECCV 2016) and VASNet.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from knapsack import generate_summary
from config import Config


def evaluate_summary(machine_summary: np.ndarray, user_summary: np.ndarray) -> float:
    """
    Compute F-score between machine summary and ONE user summary.
    
    Both inputs are binary arrays of the same length (n_frames).
    F-score = 2 * P * R / (P + R)
    
    This is the EXACT formula from the VASNet paper and Zhang et al.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    
    # Number of overlapping frames (both selected)
    overlap = np.sum(machine_summary * user_summary)
    
    # Precision = overlap / machine_selected
    machine_selected = np.sum(machine_summary)
    precision = overlap / (machine_selected + 1e-8)
    
    # Recall = overlap / user_selected  
    user_selected = np.sum(user_summary)
    recall = overlap / (user_selected + 1e-8)
    
    # F-score
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
    
    return float(f_score)


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    cfg: Config,
) -> float:
    """
    Evaluate model on dataset using the standard video summarization protocol.
    
    For each video:
    1. Predict frame importance scores
    2. Generate summary using knapsack algorithm
    3. Compare against EACH user annotation
    4. Average F-scores across users for this video
    
    Final score = mean of per-video F-scores (× 100 for percentage)
    """
    model.eval()
    video_fscores: list[float] = []

    for sample in dataset:
        # ── Get model predictions ────────────────────────────────────────
        feat = sample["features"].unsqueeze(0).to(cfg.device)

        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            pred_scores, _ = model(feat)

        pred_scores = pred_scores.cpu().numpy()  # (N_sub,)

        # ── Normalize predictions to [0, 1] ──────────────────────────────
        # Important: use min-max normalization as in paper
        score_min = pred_scores.min()
        score_max = pred_scores.max()
        if score_max - score_min > 1e-8:
            pred_scores = (pred_scores - score_min) / (score_max - score_min)
        else:
            pred_scores = np.zeros_like(pred_scores)

        # ── Generate machine summary via knapsack ────────────────────────
        machine_summary = generate_summary(
            pred_scores,
            sample["change_points"],    # (K, 2)
            int(sample["n_frames"]),    # scalar
            sample["n_frame_per_seg"],  # (K,)
            sample["picks"],            # (N_sub,)
            cfg.summary_rate,           # 0.15
        )

        # ── Get user summaries ───────────────────────────────────────────
        user_summary = sample["user_summary"]  # (n_users, n_frames)
        n_frames = user_summary.shape[1]

        # ── Ensure machine_summary matches user_summary length ───────────
        if len(machine_summary) > n_frames:
            machine_summary = machine_summary[:n_frames]
        elif len(machine_summary) < n_frames:
            # Pad with zeros
            padding = np.zeros(n_frames - len(machine_summary), dtype=np.float32)
            machine_summary = np.concatenate([machine_summary, padding])

        # ── Compute F-score against EACH user ────────────────────────────
        user_fscores: list[float] = []
        n_users = user_summary.shape[0]
        
        for u in range(n_users):
            user_u = user_summary[u]
            f = evaluate_summary(machine_summary, user_u)
            user_fscores.append(f)

        # ── Average F-score for this video ───────────────────────────────
        # Paper: "we compare with each user and take the mean"
        video_f = float(np.mean(user_fscores))
        video_fscores.append(video_f)

    # ── Final: mean across all test videos ───────────────────────────────
    mean_fscore = float(np.mean(video_fscores)) * 100.0
    
    return mean_fscore