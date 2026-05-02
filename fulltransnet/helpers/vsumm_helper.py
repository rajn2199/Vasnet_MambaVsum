# helpers/vsumm_helper.py
"""
Video summarization evaluation utilities.
Includes F1-score computation, knapsack solver, and keyshot summary generation.
"""
from typing import Iterable, List

import numpy as np


def f1_score(pred: np.ndarray, test: np.ndarray) -> float:
    """Compute F1-score on binary classification task.

    :param pred: Predicted binary label. Sized [N].
    :param test: Ground truth binary label. Sized [N].
    :return: F1-score value.
    """
    assert pred.shape == test.shape
    pred = np.asarray(pred, dtype=bool)
    test = np.asarray(test, dtype=bool)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap / pred.sum()
    recall = overlap / test.sum()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def knapsack(values: Iterable[int],
             weights: Iterable[int],
             capacity: int) -> List[int]:
    """Solve 0/1 knapsack problem.

    Tries to use Google OR-Tools if available, otherwise falls back
    to a pure Python dynamic programming implementation.

    :param values: Values of items. Sized [N].
    :param weights: Weights of items. Sized [N].
    :param capacity: Total knapsack capacity.
    :return: List of packed item indices.
    """
    values = list(values)
    weights = list(weights)
    capacity = int(capacity)

    try:
        from ortools.algorithms.pywrapknapsack_solver import KnapsackSolver
        solver = KnapsackSolver(
            KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, 'video_summ'
        )
        solver.Init(values, [weights], [capacity])
        solver.Solve()
        packed = [x for x in range(len(weights))
                  if solver.BestSolutionContains(x)]
        return packed
    except ImportError:
        # Fallback: pure Python DP knapsack
        return _knapsack_dp(values, weights, capacity)


def _knapsack_dp(values, weights, capacity):
    """Pure Python 0/1 knapsack using dynamic programming."""
    n = len(values)
    # dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i - 1][w - weights[i - 1]] + values[i - 1])

    # Backtrack to find selected items
    packed = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            packed.append(i - 1)
            w -= weights[i - 1]

    return sorted(packed)


def downsample_summ(summ: np.ndarray) -> np.ndarray:
    """Down-sample the summary by 15 times."""
    return summ[::15]


def get_keyshot_summ(pred: np.ndarray,
                     cps: np.ndarray,
                     n_frames: int,
                     nfps: np.ndarray,
                     picks: np.ndarray,
                     proportion: float = 0.15) -> np.ndarray:
    """Generate keyshot-based video summary (binary vector).

    :param pred: Predicted importance scores.
    :param cps: Change points — (n_segments, 2) array.
    :param n_frames: Original number of frames.
    :param nfps: Number of frames per segment.
    :param picks: Positions of subsampled frames in the original video.
    :param proportion: Maximum summary length ratio (default 15%).
    :return: Binary summary array of length n_frames.
    """
    assert pred.shape == picks.shape
    picks = np.asarray(picks, dtype=np.int32)

    # Upsample frame scores from subsampled sequence
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]

    # Average score per segment
    seg_scores = np.zeros(len(cps), dtype=np.int32)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())

    # Knapsack: select segments maximizing score within length budget
    limits = int(n_frames * proportion)
    packed = knapsack(seg_scores, nfps, limits)

    # Build binary summary
    summary = np.zeros(n_frames, dtype=np.bool_)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True

    return summary


def get_summ_f1score(pred_summ: np.ndarray,
                     test_summ: np.ndarray,
                     eval_metric: str = 'avg') -> float:
    """Compare predicted summary with ground truth user summaries.

    :param pred_summ: Predicted binary summary. Sized [N].
    :param test_summ: Ground truth binary summaries. Sized [U, N].
    :param eval_metric: 'avg' (mean F1 across users) or 'max' (best F1).
    :return: F1-score value.
    """
    pred_summ = np.asarray(pred_summ, dtype=bool)
    test_summ = np.asarray(test_summ, dtype=bool)
    _, n_frames = test_summ.shape

    if pred_summ.size > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size < n_frames:
        pred_summ = np.pad(pred_summ, (0, n_frames - pred_summ.size))

    f1s = [f1_score(user_summ, pred_summ) for user_summ in test_summ]

    if eval_metric == 'avg':
        final_f1 = np.mean(f1s)
    elif eval_metric == 'max':
        final_f1 = np.max(f1s)
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')

    return float(final_f1)
