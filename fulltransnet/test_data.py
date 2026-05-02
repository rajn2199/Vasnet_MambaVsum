"""Quick test: verify data loading and a single training step."""
import sys
sys.path.insert(0, '.')

from helpers import data_helper, vsumm_helper
from model.transformer import Transformer
from model.losses import compute_loss
import torch
import numpy as np

# 1) Load splits
splits = data_helper.load_yaml('./splits/tvsum.yml')
print(f"Loaded {len(splits)} splits")

s = splits[0]
print(f"Split 0: {len(s['train_keys'])} train, {len(s['test_keys'])} test")

# 2) Load one video
ds = data_helper.VideoDataset(s['test_keys'][:1])
key, seq, seqdiff, gtscore, cps, n_frames, nfps, picks, user_summary, gt_summary = ds[0]
print(f"Video: {key}")
print(f"  Features shape: {seq.shape}")
print(f"  GT score shape: {gtscore.shape}")
print(f"  Change points: {cps.shape}")
print(f"  N frames: {n_frames}")
print(f"  User summary: {user_summary.shape if user_summary is not None else 'None'}")
print(f"  GT summary: {gt_summary.shape}")

# 3) Run one training step
print("\n--- Running one training step ---")
model = Transformer(
    T=0, dim_in=1024, heads=8, enlayers=2, delayers=2,
    dim_mid=64, length=1536, window_size=16, stride=1, dff=2048
)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

keyshot_summ = vsumm_helper.get_keyshot_summ(gtscore, cps, n_frames, nfps, picks)
seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
target = vsumm_helper.downsample_summ(keyshot_summ)
summ_feature = seq_t.squeeze(0)[target]

global_idxa = cps[:, 0]
global_idxb = cps[:, 1]
idx_mid = (global_idxa + global_idxb) // 2
global_idx = np.column_stack((global_idxb, global_idxa)).flatten()
global_idx = np.concatenate((global_idx, idx_mid))

pred_summ, _, _, _ = model(seq_t, summ_feature, global_idx)
print(f"  Prediction shape: {pred_summ.shape}")

loss = compute_loss(pred_summ, target, 'bce')
print(f"  Loss: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()
print("  Backward + step OK")

print("\n=== ALL TESTS PASSED ===")
