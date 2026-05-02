# make_shots.py
"""
Compute shot boundaries (change points) for videos in a dataset.
Uses Kernel Temporal Segmentation (KTS) to detect change points.

Usage:
    python make_shots.py --dataset ../data/eccv16_dataset_tvsum_google_pool5.h5
"""
import argparse

import h5py
import numpy as np

from kts.cpd_auto import cpd_auto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    h5in = h5py.File(args.dataset, 'r')
    h5out = h5py.File(args.dataset + '.custom', 'w')

    for video_name, video_file in h5in.items():
        features = video_file['features'][...].astype(np.float32)
        gtscore = video_file['gtscore'][...].astype(np.float32)
        gtsummary = video_file['gtsummary'][...].astype(np.float32)

        seq_len = gtscore.size
        n_frames = seq_len * 15 - 1
        picks = np.arange(0, seq_len) * 15

        kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1)
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T

        n_frame_per_seg = end_frames - begin_frames

        h5out.create_dataset(video_name + '/features', data=features)
        h5out.create_dataset(video_name + '/gtscore', data=gtscore)
        h5out.create_dataset(video_name + '/change_points', data=change_points)
        h5out.create_dataset(video_name + '/n_frame_per_seg', data=n_frame_per_seg)
        h5out.create_dataset(video_name + '/n_frames', data=n_frames)
        h5out.create_dataset(video_name + '/picks', data=picks)
        h5out.create_dataset(video_name + '/gtsummary', data=gtsummary)

        print(f'  {video_name}: {seq_len} frames, {len(change_points)} segments')

    h5in.close()
    h5out.close()
    print('Done!')


if __name__ == '__main__':
    main()
