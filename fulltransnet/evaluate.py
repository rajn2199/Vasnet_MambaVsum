# evaluate.py
"""
Evaluation script for FullTransNet.
Computes F-score on validation data using keyshot-based summary evaluation.
Can be run standalone or called from the training loop.
"""
import logging
from pathlib import Path

import numpy as np
import torch

from helpers import init_helper, data_helper, vsumm_helper
from model.transformer import Transformer

logger = logging.getLogger()


def get_model(**kwargs):
    """Instantiate a FullTransNet model from keyword arguments."""
    return Transformer(
        T=0,
        dim_in=kwargs['num_feature'],
        heads=kwargs['num_head'],
        enlayers=kwargs['enlayers'],
        delayers=kwargs['delayers'],
        dim_mid=kwargs.get('dim_mid', 64),
        length=kwargs['length'],
        window_size=kwargs['window_size'],
        stride=kwargs['stride'],
        dff=kwargs['dff']
    )


def evaluate(args, model, val_loader):
    """Evaluate model on validation set.

    :param args: Parsed arguments.
    :param model: Trained FullTransNet model.
    :param val_loader: DataLoader for validation videos.
    :return: Average F-score across all validation videos.
    """
    model.eval()
    stats = data_helper.AverageMeter('fscore')

    with torch.no_grad():
        for (test_key, seq, seqdiff, gt_score, cps, n_frames,
             nfps, picks, user_summary, gt_summary) in val_loader:

            seq = torch.as_tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            # Generate ground truth keyshot summary
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gt_score, cps, n_frames, nfps, picks
            )
            target = vsumm_helper.downsample_summ(keyshot_summ)
            target_features = seq.squeeze(0)[target]

            # Compute global indices from change points
            global_idxa = cps[:, 0]
            global_idxb = cps[:, 1]
            idx_mid = (global_idxa + global_idxb) // 2
            global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
            global_idx = np.concatenate((global_idx, idx_mid))

            # Forward pass
            out, _, _, _ = model(seq, target_features, global_idx)

            # Extract predicted importance scores
            # The model outputs (n_keyframes, T) — extract max-matching scores
            pred_summ1 = torch.zeros(len(target))
            a, b = out.shape
            for j in range(b):
                column = out[:, j]
                min_value = torch.min(column)
                max_value = torch.max(column)
                for i in range(a):
                    if column[i] == max_value and max_value == torch.max(out[i, :]):
                        pred_summ1[j] = max_value
                        break
                else:
                    pred_summ1[j] = min_value

            # Use 'avg' metric for TVSum (average across annotators)
            eval_metric = 'avg' if 'tvsum' in test_key else 'max'
            keyshot_summ_pred = vsumm_helper.get_keyshot_summ(
                pred_summ1.cpu().numpy(), cps, n_frames, nfps, picks
            )
            fscore = vsumm_helper.get_summ_f1score(
                keyshot_summ_pred, user_summary, eval_metric
            )

            stats.update(fscore=fscore)

    return stats.fscore


def main():
    """Standalone evaluation entry point."""
    args = init_helper.get_arguments()
    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(args)
    model = get_model(**vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(
                args.model_dir, split_path, split_idx
            )

            state_dict = torch.load(
                str(ckpt_path), map_location=lambda storage, loc: storage
            )
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            fscore = evaluate(args, model, val_loader)
            stats.update(fscore=fscore)

            logger.info(
                f'{split_path.stem} split {split_idx}: F-score: {fscore:.4f}'
            )

        logger.info(
            f'{split_path.stem}: F-score: {stats.fscore:.4f}'
        )


if __name__ == '__main__':
    main()
