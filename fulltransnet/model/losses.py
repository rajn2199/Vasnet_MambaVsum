# model/losses.py
"""
Loss functions for FullTransNet training.
Includes: BCE, MSE, Focal, Jaccard, Tversky, and their variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def jaccard_loss(true, pred, smooth=100):
    """Jaccard (IoU) loss.

    Jaccard = |X ∩ Y| / (|X| + |Y| - |X ∩ Y|)
    We return (1 - Jaccard) * smooth so optimizer minimizes it.
    """
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
        torch.sum(true) + torch.sum(pred) - intersection + smooth
    )
    return (1 - jac) * smooth


def power_jaccard_loss(true, pred, p=1.4, smooth=100):
    """Power Jaccard loss — uses p-norm of predictions."""
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
        torch.sum(true ** p) + torch.sum(pred ** p) - intersection + smooth
    )
    return (1 - jac) * smooth


def tversky_loss(true, pred, b=0.50, smooth=100):
    """Tversky loss — generalization of Dice/Jaccard with asymmetric FP/FN weighting."""
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
        torch.sum(true * pred)
        + b * torch.sum((1 - true) * pred)
        + (1 - b) * torch.sum(true * (1 - pred))
        + smooth
    )
    return (1 - jac) * smooth


def focal_tversky(true, pred, b=0.1, smooth=100):
    """Focal Tversky loss — applies focal modulation to Tversky."""
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (
        torch.sum(true * pred)
        + b * torch.sum(true * (1 - pred))
        + (1 - b) * torch.sum((1 - true) * pred)
        + smooth
    )
    gamma = 2
    return torch.pow((1 - jac), gamma) * smooth


def one_hot_embedding(labels, num_classes):
    """Embed integer labels to one-hot vectors. (N,) -> (N, C)."""
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x, y, alpha=0.25, gamma=2, reduction='sum'):
    """Focal loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    :param x: Predicted confidence. Sized [N, D].
    :param y: Ground truth label. Sized [N].
    :param alpha: Balancing factor.
    :param gamma: Focusing parameter.
    :param reduction: 'sum', 'mean', or 'none'.
    :return: Scalar loss value.
    """
    _, num_classes = x.shape
    t = one_hot_embedding(y, num_classes).to(x.device)

    p_t = x * t + (1 - x) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()

    if reduction == 'sum':
        fl = fl.sum()
    elif reduction == 'mean':
        fl = fl.mean()
    return fl


def calc_cls_loss(pred, test, kind='focal'):
    """Compute classification loss on positive/negative samples.

    :param pred: Predicted class probabilities. Sized [N].
    :param test: Class labels (0 or 1). Sized [N].
    :param kind: 'focal' or 'cross-entropy'.
    """
    test = test.type(torch.long)
    num_pos = test.sum()

    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)

    if kind == 'focal':
        loss = focal_loss(pred, test, reduction='sum')
    elif kind == 'cross-entropy':
        loss = F.nll_loss(pred.log(), test)
    else:
        raise ValueError(f'Invalid loss type {kind}')

    loss = loss / max(num_pos, 1)
    return loss


def compute_loss(output, target, loss_name, smooth=100):
    """
    Unified loss computation for FullTransNet.

    :param output: Model output — (n_keyframes, T) softmax scores.
    :param target: Binary target array — length T, 1 where keyframes exist.
    :param loss_name: One of 'bce', 'mse', 'focal', 'jaccard', 'focal_tversky',
                      'power_jaccard', 'tversky'.
    :param smooth: Smoothing factor for Jaccard/Tversky losses.
    """
    T = len(target)
    target = torch.as_tensor(target)

    # Find indices of positive (keyframe) positions
    indices = torch.nonzero(target).squeeze().to(output.device)
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)

    # Build one-hot target matrix: (n_keyframes, T)
    target_matrix = torch.zeros(len(indices), T, device=output.device)
    for i, index in enumerate(indices):
        target_matrix[i][index] = 1

    gt_summ = target_matrix.view(-1)
    pred_summ = output.flatten()

    if loss_name == 'focal':
        loss = calc_cls_loss(pred_summ, gt_summ, 'focal')
    elif loss_name == 'bce':
        criterion = nn.BCELoss()
        loss = criterion(pred_summ, gt_summ)
    elif loss_name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(pred_summ, gt_summ)
    elif loss_name == 'jaccard':
        loss = jaccard_loss(gt_summ, pred_summ, smooth)
    elif loss_name == 'focal_tversky':
        loss = focal_tversky(gt_summ, pred_summ, smooth=smooth)
    elif loss_name == 'power_jaccard':
        loss = power_jaccard_loss(gt_summ, pred_summ)
    elif loss_name == 'tversky':
        loss = tversky_loss(gt_summ, pred_summ)
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

    return loss
