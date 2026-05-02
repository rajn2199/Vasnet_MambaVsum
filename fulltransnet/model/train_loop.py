# model/train_loop.py
"""
Training loop for FullTransNet.
Called by the main train.py entry point for each split.
"""
import logging
import torch
import numpy as np

from model.transformer import Transformer
from model.losses import compute_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper

logger = logging.getLogger()


def train(args, split, save_path, spt_idx):
    """Train FullTransNet on a single data split.

    :param args: Parsed command-line arguments.
    :param split: Dict with 'train_keys' and 'test_keys'.
    :param save_path: Path to save best checkpoint.
    :param spt_idx: Split index (for logging).
    :return: (best_fscore, model)
    """
    model = Transformer(
        T=0,
        dim_in=args.num_feature,
        heads=args.num_head,
        enlayers=args.enlayers,
        delayers=args.delayers,
        dim_mid=args.dim_mid,
        length=args.length,
        window_size=args.window_size,
        stride=args.stride,
        dff=args.dff
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'FullTransNet total params: {total_params:.4f}M')

    model = model.to(args.device)
    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    max_val_fscore = -1

    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()

        for (train_keys, seq, seqdiff, gtscore, change_points,
             n_frames, nfps, picks, user_sum, gt_summary) in train_loader:

            # Generate keyshot summary from ground truth scores
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks
            )

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            # Downsample summary to match feature sequence length
            target = vsumm_helper.downsample_summ(keyshot_summ)
            summ_feature = seq.squeeze(0)[target]

            if not target.any():
                continue

            # Compute global attention indices from change points
            global_idxa = change_points[:, 0]
            global_idxb = change_points[:, 1]
            idx_mid = (global_idxa + global_idxb) // 2
            global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
            global_idx = np.concatenate((global_idx, idx_mid))

            # Forward pass
            pred_summ, enc_self_attns, dec_self_attns, dec_enc_attns = model(
                seq, summ_feature, global_idx
            )

            # Compute loss
            loss = compute_loss(pred_summ, target, args.loss, args.smooth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_fscore = evaluate(args, model, val_loader)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        dataset = str(args.splits).split('/')[-1].split('.')[0]
        logger.info(
            f' {dataset}:{spt_idx} '
            f'Epoch: {epoch}/{args.max_epoch} '
            f'test F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}'
        )

    return max_val_fscore, model
