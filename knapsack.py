# knapsack.py
"""
Knapsack-based summary generation - Exact implementation from VASNet/Zhang et al.
Reference: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
"""
import numpy as np


def knapsack_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> list:
    """
    0/1 Knapsack via Dynamic Programming.
    
    Args:
        values: importance scores for each segment
        weights: number of frames in each segment  
        capacity: maximum frames allowed in summary
    
    Returns:
        List of selected segment indices
    """
    n = len(values)
    
    # Handle edge case
    if capacity <= 0:
        return []
    
    # DP table: dp[i][w] = max value using first i items with capacity w
    # Use int64 for weights to avoid overflow
    weights = np.array(weights, dtype=np.int64)
    capacity = int(capacity)
    
    # For memory efficiency with large capacities, use 1D DP
    dp = np.zeros(capacity + 1, dtype=np.float64)
    keep = np.zeros((n, capacity + 1), dtype=np.bool_)
    
    for i in range(n):
        wi = int(weights[i])
        vi = float(values[i])
        
        # Iterate backwards to avoid using same item twice
        for w in range(capacity, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                keep[i, w] = True
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i, w]:
            selected.append(i)
            w -= int(weights[i])
    
    return selected


def generate_summary(
    pred_scores: np.ndarray,   # (N_sub,) predicted importance scores (subsampled)
    cps: np.ndarray,           # (K, 2) change points [start, end] in original frame space
    n_frames: int,             # total number of original frames
    nfps: np.ndarray,          # (K,) number of frames per segment
    picks: np.ndarray,         # (N_sub,) indices mapping subsampled -> original frames
    proportion: float = 0.15,  # maximum summary length as fraction of video
) -> np.ndarray:
    """
    Generate video summary using 0/1 Knapsack algorithm.
    
    This follows the exact protocol from:
    - Zhang et al. "Video Summarization with Long Short-term Memory" (ECCV 2016)
    - Used by VASNet, SUM-GAN, and other video summarization papers
    
    Steps:
    1. Upsample subsampled scores to original frame space
    2. Compute mean score for each temporal segment
    3. Select segments using 0/1 Knapsack (maximize score, constrain length)
    4. Generate binary summary vector
    
    Returns:
        Binary array of shape (n_frames,) where 1 = selected frame
    """
    n_segs = len(cps)
    
    # ── Step 1: Upsample scores to original frame space ──────────────────
    # Each subsampled frame represents a range of original frames
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    picks = np.array(picks, dtype=np.int64)
    
    # Fill scores using picks as indices
    for i in range(len(pred_scores)):
        if i < len(picks):
            pos = picks[i]
            # Determine range this subsampled frame covers
            if i + 1 < len(picks):
                next_pos = picks[i + 1]
            else:
                next_pos = n_frames
            # Assign score to all frames in this range
            frame_scores[pos:next_pos] = pred_scores[i]
    
    # ── Step 2: Compute segment-level scores ─────────────────────────────
    seg_scores = np.zeros(n_segs, dtype=np.float64)
    seg_weights = np.zeros(n_segs, dtype=np.int64)
    
    for i in range(n_segs):
        start = int(cps[i, 0])
        end = int(cps[i, 1]) + 1  # +1 because end is inclusive
        end = min(end, n_frames)  # Safety check
        
        if end > start:
            seg_scores[i] = frame_scores[start:end].mean()
        seg_weights[i] = int(nfps[i])
    
    # ── Step 3: Knapsack selection ───────────────────────────────────────
    # Capacity = maximum number of frames (proportion of video length)
    capacity = int(n_frames * proportion)
    
    # Select segments that maximize total importance within capacity
    selected_segs = knapsack_dp(seg_scores, seg_weights, capacity)
    
    # ── Step 4: Generate binary summary ──────────────────────────────────
    summary = np.zeros(n_frames, dtype=np.float32)
    
    for seg_idx in selected_segs:
        start = int(cps[seg_idx, 0])
        end = int(cps[seg_idx, 1]) + 1
        end = min(end, n_frames)
        summary[start:end] = 1.0
    
    return summary