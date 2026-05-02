"""
summarize.py
Takes any .mp4 video → produces a real summarized .mp4 output.
Uses trained VASNet weights from checkpoints/

Usage:
    python summerize.py --video myvideo.mp4 --split 0
    python summerize.py --video myvideo.mp4 --split 2 --rate 0.20
"""

import argparse
import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path

from config import Config
from model import VASNet
from knapsack import generate_summary


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Extract frames from raw video
# ═══════════════════════════════════════════════════════════════

def extract_frames(video_path: str, fps: float = 2.0):
    """
    Sample frames at given FPS (same as training data preparation).
    
    Returns:
        frames_rgb        : list of (H,W,3) uint8 arrays (224×224)
        frame_times       : list of timestamps in seconds
        orig_fps          : original video FPS
        total_orig_frames : total frame count in original video
        picks             : (N_sub,) int array — maps subsampled → original frame index
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_orig_frames / orig_fps
    interval = max(1, int(round(orig_fps / fps)))

    print(f"  Video   : {video_path}")
    print(f"  Duration: {duration:.1f}s  |  "
          f"FPS: {orig_fps:.1f}  |  "
          f"Sampling: every {interval} frames (~{fps} FPS)")

    frames_rgb: list = []
    frame_times: list = []
    picks: list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            rgb = cv2.cvtColor(
                cv2.resize(frame, (224, 224)),
                cv2.COLOR_BGR2RGB
            )
            frames_rgb.append(rgb)
            frame_times.append(frame_idx / orig_fps)
            picks.append(frame_idx)
        frame_idx += 1

    cap.release()
    print(f"  Extracted {len(frames_rgb)} sampled frames "
          f"(interval={interval}, total_orig={total_orig_frames})")

    return (frames_rgb, frame_times,
            orig_fps, total_orig_frames,
            np.array(picks, dtype=np.int64))


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Extract GoogLeNet pool5 features (matches training data)
# ═══════════════════════════════════════════════════════════════

class GoogLeNetFeatureExtractor(nn.Module):
    """
    Extracts 1024-D pool5 (avgpool) features — identical to what
    the training h5 files contain (eccv16_dataset_*_google_pool5.h5).
    """
    def __init__(self):
        super().__init__()
        googlenet = models.googlenet(
            weights=models.GoogLeNet_Weights.IMAGENET1K_V1
        )
        self.features = nn.Sequential(
            googlenet.conv1,
            googlenet.maxpool1,
            googlenet.conv2,
            googlenet.conv3,
            googlenet.maxpool2,
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3,
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a,
            googlenet.inception5b,
            googlenet.avgpool,          # → (B, 1024, 1, 1)
        )
        for p in self.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def extract(self, frames_rgb: list,
                batch_size: int = 32,
                device: torch.device = torch.device("cpu")
               ) -> np.ndarray:
        self.eval().to(device)
        all_features = []
        for i in range(0, len(frames_rgb), batch_size):
            batch = frames_rgb[i : i + batch_size]
            tensors = torch.stack(
                [self.transform(f) for f in batch]
            ).to(device)
            feats = self.features(tensors)            # (B, 1024, 1, 1)
            feats = feats.squeeze(-1).squeeze(-1)     # (B, 1024)
            all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)  # (N_sub, 1024)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Run VASNet → importance scores
# ═══════════════════════════════════════════════════════════════

def get_importance_scores(
    features: np.ndarray,
    cfg: Config,
    checkpoint_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    model = VASNet(cfg).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device,
                      weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Loaded : {checkpoint_path}")
    print(f"  Ckpt F : {ckpt['fscore']:.2f}%  (epoch {ckpt['epoch']})")

    feat_t = (torch.from_numpy(features)
                   .float()
                   .unsqueeze(0)
                   .to(cfg.device))    # (1, N_sub, 1024)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            scores, attn = model(feat_t)

    return scores.cpu().numpy(), attn.cpu().numpy()   # (N_sub,), (N_sub, N_sub)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Build change points (temporal segmentation)
# ═══════════════════════════════════════════════════════════════

def build_change_points(
    picks: np.ndarray,           # (N_sub,) subsampled → original frame indices
    total_orig_frames: int,
    segment_len: int = 15,       # segment length in subsampled frames
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create uniform temporal segmentation.
    
    The paper uses KTS (Kernel Temporal Segmentation) for the benchmark datasets,
    but for new videos we use uniform segments which works reasonably well.
    
    Returns:
        cps  : (K, 2) int — [start, end] in original frame indices (inclusive)
        nfps : (K,)   int — number of original frames per segment
    """
    n_sub = len(picks)
    cps: list = []
    nfps: list = []

    i = 0
    while i < n_sub:
        # Segment spans from subsampled index i to min(i + segment_len - 1, n_sub - 1)
        end_sub = min(i + segment_len - 1, n_sub - 1)

        # Map to original frame space
        orig_start = int(picks[i])
        
        # End of segment in original space
        if end_sub + 1 < n_sub:
            # Next segment starts at picks[end_sub + 1], so this segment ends just before
            orig_end = int(picks[end_sub + 1]) - 1
        else:
            # Last segment goes to the end of the video
            orig_end = total_orig_frames - 1

        orig_end = min(orig_end, total_orig_frames - 1)
        length = orig_end - orig_start + 1

        cps.append([orig_start, orig_end])
        nfps.append(max(1, length))
        
        i = end_sub + 1

    return np.array(cps, dtype=np.int64), np.array(nfps, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Write summary video
# ═══════════════════════════════════════════════════════════════

def write_summary_video(
    input_path: str,
    output_path: str,
    summary: np.ndarray,         # binary, length = total_orig_frames
    orig_fps: float,
    total_orig_frames: int,
):
    """
    Write output video containing only selected frames.
    summary is a binary mask in original frame space from generate_summary().
    """
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, orig_fps, (w, h))

    frame_idx = 0
    written = 0

    # Ensure summary matches total frame count
    mask = np.zeros(total_orig_frames, dtype=np.int32)
    n = min(len(summary), total_orig_frames)
    mask[:n] = summary[:n].astype(np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(mask) and mask[frame_idx]:
            out.write(frame)
            written += 1
        frame_idx += 1

    cap.release()
    out.release()

    kept_sec = written / orig_fps
    total_sec = total_orig_frames / orig_fps
    print(f"\n  Summary length : {kept_sec:.1f}s  "
          f"({100 * written / max(total_orig_frames, 1):.1f}% of original)")
    print(f"  Original length: {total_sec:.1f}s")
    if written > 0:
        print(f"  Compression    : {total_sec / kept_sec:.1f}x shorter")
    print(f"  Output saved   : {output_path}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Pretty timeline visualization
# ═══════════════════════════════════════════════════════════════

def print_timeline(
    summary: np.ndarray,     # binary, original frame space
    orig_fps: float,
    scores_sub: np.ndarray,  # (N_sub,) importance scores
    picks: np.ndarray,       # (N_sub,) original frame indices
):
    """Print a visual timeline of selected segments."""
    
    def fmt(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    # Upsample scores to original frame space (same as in generate_summary)
    n_orig = len(summary)
    frame_scores = np.zeros(n_orig, dtype=np.float32)
    
    for i in range(len(scores_sub)):
        if i < len(picks):
            pos = int(picks[i])
            if i + 1 < len(picks):
                next_pos = int(picks[i + 1])
            else:
                next_pos = n_orig
            frame_scores[pos:next_pos] = scores_sub[i]

    # Collect contiguous selected segments
    events: list = []
    in_seg = False
    seg_start = 0
    
    for i, sel in enumerate(summary):
        if sel and not in_seg:
            seg_start = i
            in_seg = True
        elif not sel and in_seg:
            events.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        events.append((seg_start, len(summary) - 1))

    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │            IMPORTANT SEGMENTS FOUND              │")
    print("  ├─────────────┬─────────────┬──────────────────────┤")
    print("  │    START    │     END     │     CONFIDENCE       │")
    print("  ├─────────────┼─────────────┼──────────────────────┤")

    for s, e in events:
        t_start = s / orig_fps
        t_end = e / orig_fps
        conf = float(np.mean(frame_scores[s : e + 1]))
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        print(f"  │  {fmt(t_start)}       │  {fmt(t_end)}       │  "
              f"{bar}  {conf:.2f}  │")

    print("  └─────────────┴─────────────┴──────────────────────┘")
    print(f"  Total segments selected: {len(events)}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def find_best_checkpoint(checkpoint_dir: str) -> tuple[str, int, float]:
    """
    Scan all best_splitN.pt files and return the one with highest F-score.
    """
    best_path: str = ""
    best_split: int = -1
    best_fscore: float = -1.0

    for split_id in range(10):   # support up to 10 splits
        path = os.path.join(checkpoint_dir, f"best_split{split_id}.pt")
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            f = float(ckpt.get("fscore", -1.0))
            if f > best_fscore:
                best_fscore = f
                best_split = split_id
                best_path = path
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")

    if best_path == "":
        raise FileNotFoundError(
            f"No checkpoints found in '{checkpoint_dir}'.\n"
            "Run python train.py first!"
        )
    return best_path, best_split, best_fscore


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a video using trained VASNet"
    )
    parser.add_argument("--video", required=True,
                        help="Path to input .mp4 video")
    parser.add_argument("--split", type=int, default=None,
                        help="Force a specific checkpoint split (0–4). "
                             "Omit to auto-select the best F-score split.")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <input>_summary.mp4)")
    parser.add_argument("--rate", type=float, default=0.15,
                        help="Summary length as fraction (default 0.15 = 15%%)")
    parser.add_argument("--seg", type=int, default=15,
                        help="Segment length in sampled frames (default 15)")
    args = parser.parse_args()

    cfg = Config()
    cfg.summary_rate = args.rate

    video_path = args.video
    output_path = args.output or (Path(video_path).stem + "_summary.mp4")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ── Pick checkpoint ────────────────────────────────────
    if args.split is not None:
        ckpt_path = os.path.join(cfg.checkpoint_dir,
                                 f"best_split{args.split}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run python train.py first!"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        split_id = args.split
        fscore = float(ckpt.get("fscore", 0.0))
    else:
        ckpt_path, split_id, fscore = find_best_checkpoint(cfg.checkpoint_dir)

    print(f"\n  Using split {split_id}  |  checkpoint F-score: {fscore:.2f}%"
          f"  ({'auto-selected best' if args.split is None else 'manual'})")

    print("\n" + "═" * 54)
    print("  VASNet Video Summarizer")
    print("═" * 54)

    # ── 1. Extract frames ──────────────────────────────────
    print("\n[1/5] Extracting frames...")
    frames, times, orig_fps, total_orig, picks = extract_frames(
        video_path, fps=2.0
    )
    n_sub = len(frames)

    # ── 2. GoogLeNet pool5 features ───────────────────────
    print("\n[2/5] Extracting GoogLeNet pool5 features...")
    extractor = GoogLeNetFeatureExtractor()
    features = extractor.extract(frames, batch_size=32, device=cfg.device)
    print(f"  Feature shape: {features.shape}")   # (N_sub, 1024)

    # ── 3. VASNet scores ───────────────────────────────────
    print("\n[3/5] Running VASNet...")
    scores, _ = get_importance_scores(features, cfg, ckpt_path)
    scores = scores.astype(np.float32)
    
    # Normalize to [0, 1]
    score_min, score_max = scores.min(), scores.max()
    if score_max - score_min > 1e-8:
        scores = (scores - score_min) / (score_max - score_min)
    
    print(f"  Score range : {scores.min():.3f} – {scores.max():.3f}")
    print(f"  Mean score  : {scores.mean():.3f}")

    # ── 4. Keyshot selection ───────────────────────────────
    print("\n[4/5] Selecting keyshots (Knapsack)...")
    cps, nfps = build_change_points(picks, total_orig, segment_len=args.seg)
    budget = int(total_orig * cfg.summary_rate)
    print(f"  Segments    : {len(cps)}  |  "
          f"Budget: {budget} orig frames ({cfg.summary_rate*100:.0f}%)")

    # Generate summary using knapsack algorithm
    summary = generate_summary(
        pred_scores=scores,
        cps=cps,
        n_frames=total_orig,
        nfps=nfps,
        picks=picks,
        proportion=cfg.summary_rate,
    )
    
    print(f"  Selected {int(summary.sum())} / {total_orig} orig frames "
          f"({100 * summary.mean():.1f}%)")

    # ── 5. Write output video ──────────────────────────────
    print("\n[5/5] Writing summary video...")
    write_summary_video(
        video_path, output_path,
        summary, orig_fps, total_orig,
    )

    # ── Timeline ───────────────────────────────────────────
    print_timeline(summary, orig_fps, scores, picks)

    print("\n" + "═" * 54)
    print(f"  ✓ Done!  →  {output_path}")
    print("═" * 54 + "\n")


if __name__ == "__main__":
    main()