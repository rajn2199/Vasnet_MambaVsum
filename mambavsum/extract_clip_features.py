# extract_clip_features.py
"""
CLIP Feature Extraction for Video Summarization.

Extracts CLIP ViT-L/14 features from raw video frames and saves them
into an HDF5 file that mirrors the ECCV16 dataset format.

This replaces the 1024-d GoogLeNet pool5 features with 768-d CLIP features
while preserving all metadata (change_points, picks, user_summary, etc.)
from the original ECCV16 h5 file.

Usage:
    python extract_clip_features.py \
        --source_h5  ../data/eccv16_dataset_tvsum_google_pool5.h5 \
        --video_dir  ./videos/ \
        --output_h5  ../data/tvsum_clip_features.h5 \
        --info_tsv   ./data/ydata-tvsum50-info.tsv \
        --clip_model ViT-L/14 \
        --batch_size 64

Requirements:
    pip install torch torchvision open_clip_torch h5py opencv-python tqdm pandas
"""

import os
import argparse
import numpy as np
import h5py
import torch
import pandas as pd
from pathlib import Path

# We try OpenAI CLIP first, fall back to open_clip
try:
    import clip as openai_clip
    CLIP_BACKEND = "openai"
except ImportError:
    try:
        import open_clip
        CLIP_BACKEND = "open_clip"
    except ImportError:
        CLIP_BACKEND = None


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

def load_clip_model(model_name="ViT-L/14", device="cuda"):
    """Load CLIP model and preprocessing transform."""
    if CLIP_BACKEND == "openai":
        model, preprocess = openai_clip.load(model_name, device=device)
        model.eval()
        return model, preprocess

    elif CLIP_BACKEND == "open_clip":
        name_map = {
            "ViT-L/14": ("ViT-L-14", "openai"),
            "ViT-B/32": ("ViT-B-32", "openai"),
            "ViT-B/16": ("ViT-B-16", "openai"),
        }
        oc_name, pretrained = name_map.get(model_name, (model_name, "openai"))
        model, _, preprocess = open_clip.create_model_and_transforms(
            oc_name, pretrained=pretrained, device=device
        )
        model.eval()
        return model, preprocess

    else:
        raise ImportError(
            "Neither 'clip' (OpenAI) nor 'open_clip' is installed.\n"
            "Install one:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  OR\n"
            "  pip install open_clip_torch"
        )


# ---------------------------------------------------------------------------
# Video / frame utilities
# ---------------------------------------------------------------------------

def extract_frames_from_video(video_path, picks=None, subsample_rate=15):
    """
    Extract frames from a video file.

    If `picks` is provided, extract only those frame indices.
    Otherwise subsample every `subsample_rate` frames (ECCV16 protocol).

    Returns:
        frames       : list of PIL Images
        frame_indices: np.ndarray of extracted frame indices
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if picks is not None:
        target_indices = set(int(p) for p in picks)
    else:
        target_indices = set(range(0, total_frames, subsample_rate))

    frame_indices = sorted(target_indices)
    idx_set = set(frame_indices)

    frame_id = 0
    collected = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in idx_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            collected[frame_id] = Image.fromarray(rgb)
        frame_id += 1

    cap.release()

    ordered_frames = [collected[i] for i in frame_indices if i in collected]
    valid_indices  = np.array([i for i in frame_indices if i in collected])

    return ordered_frames, valid_indices


@torch.no_grad()
def extract_clip_features_batch(model, preprocess, frames, batch_size=64, device="cuda"):
    """
    Extract L2-normalised CLIP visual features from a list of PIL Images.

    Returns:
        features: np.ndarray of shape (N, clip_dim)
    """
    all_features = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        tensors = torch.stack([preprocess(f) for f in batch]).to(device)
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalise
        all_features.append(feats.cpu().float().numpy())
    return np.concatenate(all_features, axis=0)


# ---------------------------------------------------------------------------
# Filename resolution
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".mov"]


def find_video_by_name(video_dir, name):
    """
    Exact stem lookup first (e.g. '_xMr-HKMfVA.mp4'), then partial match.
    `name` is expected to be the bare YouTube ID (no extension).
    """
    for ext in VIDEO_EXTENSIONS:
        candidate = Path(video_dir) / (name + ext)
        if candidate.exists():
            return candidate
    # Fallback: substring match (handles filenames like 'tvsum_<id>.mp4')
    for p in Path(video_dir).iterdir():
        if p.suffix.lower() in VIDEO_EXTENSIONS and name in p.stem:
            return p
    return None


# ---------------------------------------------------------------------------
# TSV-based mapping  (PRIMARY — authoritative)
# ---------------------------------------------------------------------------

def build_map_from_tsv(h5_keys, video_dir, info_tsv_path):
    """
    Build {h5_key: Path | None} using ydata-tvsum50-info.tsv.

    The TSV ships with the TVSum dataset and lists all 50 YouTube IDs in
    the exact order the h5 file was created.  Column layout:
        col 0 : video category code
        col 1 : YouTube video ID          ← this is what we need
        col 2 : video title
        col 3 : URL
        col 4 : length (MM:SS)

    h5 keys (video_1 … video_50) are sorted numerically and zipped
    with the TSV rows — that gives the guaranteed correct correspondence.
    """
    info_tsv_path = Path(info_tsv_path)
    if not info_tsv_path.exists():
        raise FileNotFoundError(f"Info TSV not found: {info_tsv_path}")

    # The file has no header row in some distributions; handle both cases.
    df = pd.read_csv(info_tsv_path, sep="\t", header=None)

    # Detect whether first row looks like a header (non-ID text in col 1)
    # YouTube IDs are 11-character alphanumeric strings.
    first_val = str(df.iloc[0, 1])
    if len(first_val) != 11:          # Likely a header row
        df = pd.read_csv(info_tsv_path, sep="\t", header=0)

    # Column 1 = YouTube ID (see dataset README)
    youtube_ids = df.iloc[:, 1].astype(str).str.strip().tolist()

    if len(youtube_ids) != 50:
        raise ValueError(
            f"Expected 50 rows in info TSV, found {len(youtube_ids)}. "
            "Check the file path / format."
        )

    # Sort h5 keys numerically: video_1, video_2, …, video_50
    def numeric_key(k):
        try:
            return int(k.split("_")[-1])
        except ValueError:
            return 0

    sorted_keys = sorted(h5_keys, key=numeric_key)

    if len(sorted_keys) != 50:
        print(
            f"  WARNING: Expected 50 h5 keys, found {len(sorted_keys)}. "
            "Mapping as many as possible."
        )

    print("\n  TSV-based key → YouTube ID → video file mapping:")
    print(f"  {'H5 Key':<12} {'YouTube ID':<14} {'Status'}")
    print("  " + "-" * 50)

    mapping = {}
    for key, yt_id in zip(sorted_keys, youtube_ids):
        video_path = find_video_by_name(video_dir, yt_id)
        mapping[key] = video_path
        status = f"✓  {video_path.name}" if video_path else "✗  MISSING"
        print(f"  {key:<12} {yt_id:<14} {status}")

    # Any keys beyond the 50 TSV rows get None
    for key in sorted_keys[len(youtube_ids):]:
        mapping[key] = None
        print(f"  {key:<12} {'(no TSV row)':<14} ✗  MISSING")

    found = sum(1 for v in mapping.values() if v is not None)
    print(f"\n  Mapped: {found} / {len(mapping)} videos found on disk.\n")

    return mapping


# ---------------------------------------------------------------------------
# Fallback: embedded video_name field in h5
# ---------------------------------------------------------------------------

def get_video_name_from_h5(video_data):
    """Try to read an embedded video filename from the h5 group."""
    for field in ("video_name", "name", "vid_name"):
        if field not in video_data:
            continue
        raw = video_data[field][()]
        if isinstance(raw, (bytes, np.bytes_)):
            return raw.decode("utf-8").strip()
        if isinstance(raw, np.ndarray):
            item = raw.item()
            if isinstance(item, (bytes, np.bytes_)):
                return item.decode("utf-8").strip()
            return str(item).strip()
        return str(raw).strip()
    return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features_for_dataset(
    source_h5_path,
    video_dir,
    output_h5_path,
    info_tsv_path=None,
    clip_model_name="ViT-L/14",
    batch_size=64,
    device="cuda",
):
    """
    Extract CLIP features for all videos in an ECCV16-format dataset.

    Mapping strategy (in priority order):
      1. ydata-tvsum50-info.tsv   → TSV row order  (AUTHORITATIVE ✓)
      2. Embedded 'video_name'    → name-based lookup
      [alphabetical fallback removed — unreliable]
    """
    from tqdm import tqdm

    print(f"Loading CLIP model : {clip_model_name}")
    model, preprocess = load_clip_model(clip_model_name, device)

    print(f"\nSource h5 : {source_h5_path}")
    print(f"Video dir : {video_dir}")
    print(f"Output h5 : {output_h5_path}")
    if info_tsv_path:
        print(f"Info TSV  : {info_tsv_path}")

    with h5py.File(source_h5_path, "r") as src, \
         h5py.File(output_h5_path, "w") as dst:

        h5_keys = list(src.keys())

        # ---- Choose mapping strategy ----------------------------------------
        if info_tsv_path and Path(info_tsv_path).exists():
            print("\n  [Strategy] TSV-based mapping (authoritative).")
            key_to_path = build_map_from_tsv(h5_keys, video_dir, info_tsv_path)

        else:
            # Try embedded video_name as second option
            first_key = h5_keys[0]
            embedded_name = get_video_name_from_h5(src[first_key])

            if embedded_name is not None:
                print("\n  [Strategy] Embedded video_name field in h5.")
                key_to_path = {}
                for key in h5_keys:
                    name = get_video_name_from_h5(src[key])
                    key_to_path[key] = find_video_by_name(video_dir, name) if name else None
            else:
                raise RuntimeError(
                    "No --info_tsv provided and h5 contains no embedded video_name.\n"
                    "Please supply --info_tsv ydata-tvsum50-info.tsv for reliable mapping."
                )
        # ---------------------------------------------------------------------

        print(f"Processing {len(h5_keys)} videos...\n")
        found_count   = 0
        missing_count = 0

        for key in tqdm(h5_keys, desc="Extracting CLIP features"):
            video_data = src[key]
            picks = np.array(video_data["picks"][()])
            video_path = key_to_path.get(key)

            if video_path is not None and Path(video_path).exists():
                found_count += 1
                try:
                    frames, _ = extract_frames_from_video(video_path, picks=picks)
                    if len(frames) > 0:
                        features = extract_clip_features_batch(
                            model, preprocess, frames, batch_size, device
                        )
                    else:
                        tqdm.write(
                            f"  Warning: 0 frames extracted for {key} "
                            f"({Path(video_path).name}) — using zeros"
                        )
                        features = np.zeros((len(picks), 768), dtype=np.float32)
                except Exception as e:
                    tqdm.write(
                        f"  Error processing {key} ({Path(video_path).name}): "
                        f"{e} — using zeros"
                    )
                    features = np.zeros((len(picks), 768), dtype=np.float32)
            else:
                missing_count += 1
                tqdm.write(f"  Not found: {key} — using zeros")
                features = np.zeros((len(picks), 768), dtype=np.float32)

            # Align feature count to picks count (should already match)
            if len(features) < len(picks):
                pad = np.zeros(
                    (len(picks) - len(features), features.shape[1]), dtype=np.float32
                )
                features = np.concatenate([features, pad], axis=0)
            elif len(features) > len(picks):
                features = features[: len(picks)]

            # Write output group — features replaced, all metadata preserved
            grp = dst.create_group(key)
            grp.create_dataset("features", data=features.astype(np.float32))
            for meta_key in video_data.keys():
                if meta_key != "features":
                    grp.create_dataset(meta_key, data=video_data[meta_key][()])

    print(f"\n[Done] CLIP features saved to : {output_h5_path}")
    print(f"  Feature dim  : 768  (CLIP {clip_model_name})")
    print(f"  Found        : {found_count} / {len(h5_keys)}")
    print(f"  Missing      : {missing_count} / {len(h5_keys)}")
    if missing_count > 0:
        print(
            "  Missing videos received all-zero features.\n"
            "  Double-check --video_dir and ensure filenames contain YouTube IDs."
        )


# ---------------------------------------------------------------------------
# Synthetic fallback (no raw videos — for unit testing only)
# ---------------------------------------------------------------------------

def generate_synthetic_clip_features(source_h5_path, output_h5_path):
    """
    Generate synthetic CLIP features by projecting GoogLeNet features.
    TESTING ONLY — use real CLIP features for actual experiments.
    """
    print("=" * 60)
    print("  GENERATING SYNTHETIC CLIP FEATURES (testing only)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    projection = rng.standard_normal((1024, 768)).astype(np.float32)
    projection /= np.linalg.norm(projection, axis=1, keepdims=True)

    with h5py.File(source_h5_path, "r") as src, \
         h5py.File(output_h5_path, "w") as dst:

        keys = list(src.keys())
        print(f"Processing {len(keys)} videos...")

        for key in keys:
            video_data = src[key]
            gn_feats = np.array(video_data["features"][()], dtype=np.float32)
            clip_feats = gn_feats @ projection
            norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
            clip_feats /= norms + 1e-8

            grp = dst.create_group(key)
            grp.create_dataset("features", data=clip_feats)
            for meta_key in video_data.keys():
                if meta_key != "features":
                    grp.create_dataset(meta_key, data=video_data[meta_key][()])

    print(f"[Done] Synthetic CLIP features saved to: {output_h5_path}")


# ---------------------------------------------------------------------------
# Inspect utility
# ---------------------------------------------------------------------------

def inspect_h5(h5_path, info_tsv_path=None):
    """Print mapping for the first few groups — useful for sanity checking."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
        print(f"Total groups : {len(keys)}")
        print(f"Fields in '{keys[0]}': {list(f[keys[0]].keys())}")
        print(f"Feature shape: {f[keys[0]]['features'][()].shape}")

    if info_tsv_path and Path(info_tsv_path).exists():
        df = pd.read_csv(info_tsv_path, sep="\t", header=None)
        first_val = str(df.iloc[0, 1])
        if len(first_val) != 11:
            df = pd.read_csv(info_tsv_path, sep="\t", header=0)
        youtube_ids = df.iloc[:, 1].astype(str).str.strip().tolist()
        print(f"\nFirst 5 TSV rows (col 1 = YouTube ID):")
        for i, yt_id in enumerate(youtube_ids[:5]):
            print(f"  {keys[i]:10s}  →  {yt_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features for video summarization (ECCV16 h5 format)"
    )
    parser.add_argument(
        "--source_h5", type=str,
        default="../data/eccv16_dataset_tvsum_google_pool5.h5",
        help="Path to the original ECCV16-format HDF5 file.",
    )
    parser.add_argument(
        "--video_dir", type=str,
        default="./videos/",
        help="Directory containing downloaded TVSum video files.",
    )
    parser.add_argument(
        "--output_h5", type=str,
        default="../data/tvsum_clip_features.h5",
        help="Output HDF5 path for CLIP features.",
    )
    parser.add_argument(
        "--info_tsv", type=str,
        default="./data/ydata-tvsum50-info.tsv",
        help=(
            "Path to ydata-tvsum50-info.tsv (ships with TVSum dataset). "
            "Provides the authoritative h5-key → YouTube-ID mapping. "
            "STRONGLY recommended; script errors without it if h5 has no embedded video_name."
        ),
    )
    parser.add_argument(
        "--clip_model", type=str, default="ViT-L/14",
        help="CLIP model variant (ViT-L/14, ViT-B/32, ViT-B/16).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for CLIP inference.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic features from GoogLeNet (no raw videos needed).",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print h5 metadata and TSV mapping for the first few videos, then exit.",
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_h5(args.source_h5, args.info_tsv)

    elif args.synthetic:
        generate_synthetic_clip_features(args.source_h5, args.output_h5)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        extract_features_for_dataset(
            source_h5_path=args.source_h5,
            video_dir=args.video_dir,
            output_h5_path=args.output_h5,
            info_tsv_path=args.info_tsv,
            clip_model_name=args.clip_model,
            batch_size=args.batch_size,
            device=device,
        )