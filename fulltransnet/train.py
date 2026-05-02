# train.py
"""
Main training entry point for FullTransNet.
Trains on TVSum dataset using 5-fold cross-validation splits.

Usage:
    python train.py
    python train.py --splits ./splits/tvsum.yml --max-epoch 300
    python train.py --device cpu --max-epoch 10
"""
import logging
from pathlib import Path

from model.train_loop import train as train_model
from helpers import init_helper, data_helper
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger()

TRAINER = {
    'encoder-decoder': train_model
}


def get_trainer(model_type):
    """Get the trainer function for the given model type."""
    assert model_type in TRAINER, f'Unknown model type: {model_type}'
    return TRAINER[model_type]


def main():
    args = init_helper.get_arguments()

    # Auto-fallback to CPU if CUDA not available
    if args.device == 'cuda' and not __import__('torch').cuda.is_available():
        print('CUDA not available, falling back to CPU')
        args.device = 'cpu'

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(args)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_helper.get_ckpt_dir(model_dir).mkdir(parents=True, exist_ok=True)

    trainer = get_trainer(args.model)

    # Save configuration
    data_helper.dump_yaml(vars(args), model_dir / 'args.yml')

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        results = {}
        stats = data_helper.AverageMeter('fscore')
        test_mean_fs = []

        for split_idx, split in enumerate(splits):
            logger.info(
                f'Start training on {split_path.stem}: split {split_idx}'
            )

            ckpt_path = data_helper.get_ckpt_path(
                model_dir, split_path, split_idx
            )

            fscore, model = trainer(args, split, ckpt_path, split_idx)
            test_mean_fs.append(fscore)

            stats.update(fscore=fscore)
            results[f'split{split_idx}'] = float(fscore)

        results['mean'] = float(stats.fscore)
        data_helper.dump_yaml(results, model_dir / f'{split_path.stem}.yml')

        logger.info(
            f'Training done on {split_path.stem}. '
            f'F-score: {stats.fscore:.4f}'
        )


if __name__ == '__main__':
    main()
