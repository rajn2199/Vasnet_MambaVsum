# helpers/init_helper.py
"""
Argument parsing, logger initialization, and random seed setup
for FullTransNet training and evaluation.
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(log_dir: str, log_file: str):
    """Initialize logger with file and console handlers."""
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser():
    """Build argument parser with all FullTransNet hyperparameters."""
    parser = argparse.ArgumentParser(
        description='FullTransNet: Full Transformer with Local-Global Attention '
                    'for Video Summarization'
    )

    # Model type
    parser.add_argument('--model', type=str, default='encoder-decoder')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))

    # Random seed
    parser.add_argument('--seed', type=int, default=2021)

    # Model architecture
    parser.add_argument('--length', type=int, default=1536,
                        help='Maximum sequence length (frames are padded to this)')
    parser.add_argument('--window-size', type=int, default=16,
                        help='Local attention window size')
    parser.add_argument('--dff', type=int, default=2048,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--stride', type=int, default=1,
                        help='Attention stride')
    parser.add_argument('--num_head', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_feature', type=int, default=1024,
                        help='Input feature dimension (GoogLeNet pool5)')
    parser.add_argument('--dim_mid', type=int, default=64,
                        help='Transformer hidden dimension')
    parser.add_argument('--enlayers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--delayers', type=int, default=6,
                        help='Number of decoder layers')

    # Loss
    parser.add_argument('--loss', type=str, default='bce',
                        choices=('bce', 'focal', 'mse', 'focal_tversky',
                                 'jaccard', 'power_jaccard', 'tversky'))
    parser.add_argument('--smooth', type=int, default=100,
                        choices=(50, 100, 200))

    # Data — default to TVSum from parent data folder
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['./splits/tvsum.yml'])

    # Training
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Checkpoint / logging
    parser.add_argument('--model-dir', type=str, default='./model_save/tvsum')
    parser.add_argument('--log-file', type=str, default='log_tvsum.txt')

    return parser


def get_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = get_parser()
    args = parser.parse_args()
    return args
