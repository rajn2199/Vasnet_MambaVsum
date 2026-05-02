# knapsack.py
"""
Knapsack-based summary generation — standard protocol.
Ref: Zhang et al. (ECCV 2016), VASNet, SUM-GAN.
"""
import numpy as np


def knapsack_dp(values, weights, capacity):
    """0/1 Knapsack via Dynamic Programming."""
    n = len(values)
    if capacity <= 0:
        return []

    weights = np.array(weights, dtype=np.int64)
    capacity = int(capacity)

    dp = np.zeros(capacity + 1, dtype=np.float64)
    keep = np.zeros((n, capacity + 1), dtype=np.bool_)

    for i in range(n):
        wi = int(weights[i])
        vi = float(values[i])
        for w in range(capacity, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                keep[i, w] = True

    selected = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i, w]:
            selected.append(i)
            w -= int(weights[i])

    return selected


def generate_summary(pred_scores, cps, n_frames, nfps, picks, proportion=0.15):
    """
    Generate video summary using 0/1 Knapsack algorithm.
    Returns: binary array of shape (n_frames,) where 1 = selected frame
    """
    n_segs = len(cps)

    # Step 1: Upsample scores to original frame space
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    picks = np.array(picks, dtype=np.int64)

    for i in range(len(pred_scores)):
        if i < len(picks):
            pos = picks[i]
            next_pos = picks[i + 1] if i + 1 < len(picks) else n_frames
            frame_scores[pos:next_pos] = pred_scores[i]

    # Step 2: Compute segment-level scores
    seg_scores = np.zeros(n_segs, dtype=np.float64)
    seg_weights = np.zeros(n_segs, dtype=np.int64)

    for i in range(n_segs):
        start = int(cps[i, 0])
        end = min(int(cps[i, 1]) + 1, n_frames)
        if end > start:
            seg_scores[i] = frame_scores[start:end].mean()
        seg_weights[i] = int(nfps[i])

    # Step 3: Knapsack selection
    capacity = int(n_frames * proportion)
    selected_segs = knapsack_dp(seg_scores, seg_weights, capacity)

    # Step 4: Generate binary summary
    summary = np.zeros(n_frames, dtype=np.float32)
    for seg_idx in selected_segs:
        start = int(cps[seg_idx, 0])
        end = min(int(cps[seg_idx, 1]) + 1, n_frames)
        summary[start:end] = 1.0

    return summary
