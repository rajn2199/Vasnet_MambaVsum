# summarize_video.py
"""
MambaVSum — End-to-End Video Summarization Inference Script.

Takes a raw video file as input and produces a summarized video as output.

Pipeline:
  1. Decode video frames (OpenCV)
  2. Extract CLIP visual features (ViT-L/14)
  3. Extract audio features (mel-spectrogram → 128-d)
  4. Detect scene boundaries via KTS (Kernel Temporal Segmentation)
  5. Run MambaVSum model → frame importance scores
  6. Select best segments via 0/1 Knapsack (≤15% of original duration)
  7. Write summarized video (selected segments concatenated)

Usage:
    python summarize_video.py --video path/to/input.mp4
    python summarize_video.py --video path/to/input.mp4 --output summary.mp4
    python summarize_video.py --video path/to/input.mp4 --ratio 0.20  # 20% summary
    python summarize_video.py --video path/to/input.mp4 --checkpoint ./checkpoints/best_split0.pt

Requirements:
    pip install torch torchvision opencv-python numpy tqdm
    pip install open_clip_torch   # For CLIP features (optional, has fallback)
    pip install torchaudio         # For audio features (optional, has fallback)
"""

import argparse
import os
import sys
import math
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

# ── Add project root to path ────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from model.mambavsum import MambaVSum
from knapsack import generate_summary


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1: VIDEO DECODING
# ═══════════════════════════════════════════════════════════════════════

def decode_video(video_path, subsample_rate=15):
    """
    Decode video into subsampled frames (PIL Images) + metadata.

    The ECCV16 protocol subsamples every 15th frame. We follow the same
    convention so the model sees data in the same format as training.

    Returns:
        frames:      list of PIL Images (subsampled)
        all_frames:  list of np.ndarray BGR frames (ALL frames, for output)
        picks:       np.ndarray of selected frame indices
        fps:         original video FPS
        n_frames:    total frame count
    """
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {Path(video_path).name}")
    print(f"  Frames: {n_frames} | FPS: {fps:.1f} | Resolution: {width}x{height}")
    print(f"  Duration: {n_frames / fps:.1f}s")

    # Determine which frames to subsample
    pick_indices = set(range(0, n_frames, subsample_rate))
    picks = sorted(pick_indices)

    frames_pil = {}
    frame_idx = 0

    print(f"  Decoding {n_frames} frames (subsampling every {subsample_rate})...")
    pbar = tqdm(total=n_frames, desc="  Decoding", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in pick_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_pil[frame_idx] = Image.fromarray(rgb)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    ordered_frames = [frames_pil[i] for i in picks if i in frames_pil]
    valid_picks = np.array([i for i in picks if i in frames_pil], dtype=np.int32)

    print(f"  Subsampled frames: {len(ordered_frames)}")
    return ordered_frames, valid_picks, fps, n_frames, width, height


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2: CLIP FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_clip_features(frames, device="cpu", batch_size=32):
    """
    Extract CLIP ViT-L/14 features from a list of PIL Images.

    Falls back to random projections from pixel means if CLIP is unavailable.

    Returns:
        features: np.ndarray (N, 768)
    """
    try:
        # Try OpenAI CLIP
        try:
            import clip as openai_clip
            print("  Using OpenAI CLIP (ViT-L/14)...")
            model, preprocess = openai_clip.load("ViT-L/14", device=device)
        except ImportError:
            import open_clip
            print("  Using open_clip (ViT-L-14)...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )

        model.eval()
        all_features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(frames), batch_size),
                          desc="  CLIP features", unit="batch"):
                batch = frames[i:i + batch_size]
                tensors = torch.stack([preprocess(f) for f in batch]).to(device)
                feats = model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_features.append(feats.cpu().float().numpy())

        return np.concatenate(all_features, axis=0)

    except ImportError:
        print("  WARNING: CLIP not available. Using pixel-based fallback features.")
        print("  Install with: pip install open_clip_torch")
        print("  Results will be degraded without CLIP.")

        # Fallback: simple pixel statistics → 768-d
        features = np.zeros((len(frames), 768), dtype=np.float32)
        for i, frame in enumerate(frames):
            arr = np.array(frame).astype(np.float32) / 255.0
            # Compute spatial statistics as crude features
            mean_rgb = arr.mean(axis=(0, 1))  # (3,)
            std_rgb = arr.std(axis=(0, 1))    # (3,)
            # Tile to fill 768 dims
            base = np.concatenate([mean_rgb, std_rgb])  # (6,)
            features[i] = np.tile(base, 768 // 6 + 1)[:768]

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        return features


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3: AUDIO FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_audio_features(video_path, picks, n_frames, fps, audio_dim=128):
    """
    Extract 128-d audio features aligned with visual frame picks.

    Self-contained implementation — doesn't import extract_audio_features.py
    (which has a pandas dependency that can cause version conflicts).

    Falls back to zero features if audio extraction fails.

    Returns:
        features: np.ndarray (N, 128)
    """
    try:
        print("  Extracting audio features...")

        # --- Step A: Extract audio waveform ---
        sr = 16000
        waveform = None

        try:
            import torchaudio
            wav, orig_sr = torchaudio.load(str(video_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if orig_sr != sr:
                resampler = torchaudio.transforms.Resample(orig_sr, sr)
                wav = resampler(wav)
            waveform = wav.squeeze(0).numpy()
        except Exception:
            # Fallback: ffmpeg
            import subprocess, tempfile, wave
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    [ffmpeg_exe, "-i", str(video_path),
                     "-ac", "1", "-ar", str(sr), "-f", "wav", "-y", tmp_path],
                    capture_output=True, check=True,
                )
                with wave.open(tmp_path, "rb") as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    waveform = samples / 32768.0
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if waveform is None:
            raise RuntimeError("Could not extract audio")

        print(f"  Audio: {len(waveform) / sr:.1f}s at {sr}Hz")

        # --- Step B: Compute mel spectrogram ---
        n_mels = 64
        hop_length = 160
        n_fft = 400

        try:
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            )
            mel = mel_transform(torch.from_numpy(waveform).unsqueeze(0))
            mel = torch.log(mel + 1e-8)
            mel_spec = mel.squeeze(0).T.numpy()  # (n_time, n_mels)
        except Exception:
            # Pure numpy STFT fallback
            n_audio_frames = 1 + (len(waveform) - n_fft) // hop_length
            mel_spec = np.zeros((n_audio_frames, n_mels), dtype=np.float32)
            window = np.hanning(n_fft).astype(np.float32)
            for i in range(n_audio_frames):
                start = i * hop_length
                frame = waveform[start:start + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
                # Simple mel bins
                freq_bins = len(spectrum)
                mel_points = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
                hz_points = 700 * (10 ** (mel_points / 2595) - 1)
                bin_pts = np.clip((hz_points * n_fft / sr).astype(int), 0, freq_bins - 1)
                for m in range(n_mels):
                    s, e = bin_pts[m], bin_pts[m + 2]
                    if e > s:
                        mel_spec[i, m] = np.log(np.mean(spectrum[s:e + 1]) + 1e-8)

        total_mel_frames = mel_spec.shape[0]
        audio_fps = sr / hop_length  # 100 fps

        # --- Step C: Align audio with visual picks ---
        features = np.zeros((len(picks), audio_dim), dtype=np.float32)
        window_size = int(audio_fps * 0.5)

        for i, pick_idx in enumerate(picks):
            time_sec = pick_idx / max(fps, 1)
            mel_center = int(time_sec * audio_fps)
            mel_start = max(0, mel_center - window_size // 2)
            mel_end = min(total_mel_frames, mel_center + window_size // 2)

            if mel_start < mel_end and mel_start < total_mel_frames:
                segment = mel_spec[mel_start:mel_end]
                mean_feat = segment.mean(axis=0)
                std_feat = segment.std(axis=0)
                combined = np.concatenate([mean_feat, std_feat])
                features[i] = combined[:audio_dim] if len(combined) >= audio_dim else \
                    np.pad(combined, (0, max(0, audio_dim - len(combined))))

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        print(f"  Audio features: {features.shape}")
        return features

    except Exception as e:
        print(f"  WARNING: Audio extraction failed: {e}")
        print(f"  Using zero audio features (visual-only mode)")
        return np.zeros((len(picks), audio_dim), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 4: CHANGE POINT DETECTION (KTS)
# ═══════════════════════════════════════════════════════════════════════

def detect_change_points(features, n_frames, picks, max_segments=20):
    """
    Detect scene boundaries using Kernel Temporal Segmentation.

    If KTS modules are unavailable, falls back to uniform segmentation.

    Returns:
        change_points: np.ndarray (K, 2) — segment [start, end] pairs
        n_frame_per_seg: np.ndarray (K,) — frames per segment
    """
    try:
        # Try to import KTS from the fulltransnet sibling directory
        kts_dir = SCRIPT_DIR.parent / "fulltransnet" / "kts"
        if kts_dir.exists():
            sys.path.insert(0, str(kts_dir.parent))
            from kts.cpd_auto import cpd_auto
            from kts.cpd_nonlin import cpd_nonlin

            print("  Running KTS change point detection...")
            # Compute kernel matrix
            K = features @ features.T
            # Auto change point detection
            cps, _ = cpd_auto(K, ncp=max_segments, vmax=1)
            cps = np.concatenate(([0], cps, [len(features) - 1]))
        else:
            raise ImportError("KTS not found")

    except Exception as e:
        print(f"  KTS unavailable ({e}), using uniform segmentation...")
        # Uniform segmentation fallback
        n_segs = min(max_segments, max(2, len(features) // 30))
        seg_size = len(features) // n_segs
        cps = np.arange(0, len(features), seg_size)
        if cps[-1] != len(features) - 1:
            cps = np.append(cps, len(features) - 1)

    # Convert change points to (start, end) pairs in original frame space
    change_points = []
    n_frame_per_seg = []

    for i in range(len(cps) - 1):
        start_pick = cps[i]
        end_pick = cps[i + 1]

        # Map back to original frame indices
        start_frame = int(picks[min(start_pick, len(picks) - 1)])
        end_frame = int(picks[min(end_pick, len(picks) - 1)])

        if end_frame > start_frame:
            change_points.append([start_frame, end_frame])
            n_frame_per_seg.append(end_frame - start_frame)

    change_points = np.array(change_points, dtype=np.int32)
    n_frame_per_seg = np.array(n_frame_per_seg, dtype=np.int32)

    print(f"  Detected {len(change_points)} segments")
    return change_points, n_frame_per_seg


# ═══════════════════════════════════════════════════════════════════════
#  STEP 5: LOAD MODEL & RUN INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, cfg, device):
    """Load a trained MambaVSum model from checkpoint."""
    print(f"\n  Loading model from: {checkpoint_path}")
    model = MambaVSum(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location=device,
                            weights_only=False)

    # Handle both formats: full training checkpoint or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "?")
        fscore = checkpoint.get("fscore", "?")
        print(f"  Checkpoint from epoch {epoch}, F-score: {fscore}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model


@torch.no_grad()
def predict_scores(model, visual_features, audio_features, change_points, cfg):
    """
    Run MambaVSum forward pass → frame importance scores.

    Returns:
        scores: np.ndarray (N,) in [0, 1]
    """
    visual = torch.from_numpy(visual_features).float().unsqueeze(0).to(cfg.device)

    audio = None
    if cfg.feature_mode == "multimodal" and audio_features is not None:
        audio = torch.from_numpy(audio_features).float().unsqueeze(0).to(cfg.device)

    with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp and cfg.device.type == "cuda"):
        scores, _ = model(visual, audio, change_points)

    scores = scores.cpu().numpy()

    # Normalize to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min > 1e-8:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.ones_like(scores) * 0.5

    return scores


# ═══════════════════════════════════════════════════════════════════════
#  STEP 6: GENERATE SUMMARY & WRITE VIDEO
# ═══════════════════════════════════════════════════════════════════════

def write_summary_video(video_path, output_path, summary_mask, fps):
    """
    Read original video and write only the frames where summary_mask == 1.
    Uses ffmpeg to seamlessly preserve and concatenate both video and audio.
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"

    # 1. Convert frame mask to contiguous time segments
    segments = []
    in_segment = False
    start_frame = 0
    for i in range(len(summary_mask)):
        if summary_mask[i] > 0.5 and not in_segment:
            in_segment = True
            start_frame = i
        elif summary_mask[i] < 0.5 and in_segment:
            in_segment = False
            segments.append((start_frame, i))
    if in_segment:
        segments.append((start_frame, len(summary_mask)))

    total_selected = sum(e - s for s, e in segments)
    n_frames = len(summary_mask)

    print(f"\n  Writing summary video (with audio)...")
    print(f"  Selected {total_selected} / {n_frames} frames "
          f"({100 * total_selected / n_frames:.1f}%) across {len(segments)} segments")

    if not segments:
        print("  WARNING: No segments selected.")
        return output_path

    # 2. Build the ffmpeg filter_complex string
    filter_complex = ""
    concat_inputs = ""
    for idx, (start_f, end_f) in enumerate(segments):
        start_t = start_f / fps
        end_t = end_f / fps
        filter_complex += f"[0:v]trim=start={start_t:.3f}:end={end_t:.3f},setpts=PTS-STARTPTS[v{idx}]; "
        filter_complex += f"[0:a]atrim=start={start_t:.3f}:end={end_t:.3f},asetpts=PTS-STARTPTS[a{idx}]; "
        concat_inputs += f"[v{idx}][a{idx}]"
    
    filter_complex += f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]"

    cmd = [
        ffmpeg_exe, "-y", "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: FFmpeg failed: {e.stderr.decode('utf-8', errors='ignore')}")
        print("  Ensure the input video has an audio track.")

    duration_original = n_frames / fps
    duration_summary = total_selected / fps

    print(f"\n  [OK] Summary video saved: {output_path}")
    print(f"  Original duration : {duration_original:.1f}s")
    print(f"  Summary duration  : {duration_summary:.1f}s")
    print(f"  Compression ratio : {duration_summary / duration_original * 100:.1f}%")

    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def summarize(args):
    """Full end-to-end video summarization pipeline."""

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_summary.mp4"

    device = torch.device(args.device)
    print("=" * 60)
    print("  MambaVSum — Video Summarization")
    print("=" * 60)
    print(f"  Input  : {video_path}")
    print(f"  Output : {output_path}")
    print(f"  Device : {device}")
    print(f"  Ratio  : {args.ratio * 100:.0f}% of original")
    print("=" * 60)

    # ── Step 1: Decode video ──────────────────────────────────────
    print("\n[1/6] Decoding video...")
    frames, picks, fps, n_frames, w, h = decode_video(video_path)

    # ── Step 2: Extract CLIP features ─────────────────────────────
    print("\n[2/6] Extracting visual features...")
    visual_features = extract_clip_features(frames, device=str(device))
    print(f"  Visual features: {visual_features.shape}")

    # ── Step 3: Extract audio features ────────────────────────────
    print("\n[3/6] Extracting audio features...")
    audio_features = extract_audio_features(
        str(video_path), picks, n_frames, fps
    )

    # ── Step 4: Detect change points ──────────────────────────────
    print("\n[4/6] Detecting scene boundaries...")
    change_points, n_frame_per_seg = detect_change_points(
        visual_features, n_frames, picks
    )

    # ── Step 5: Load model & predict scores ───────────────────────
    print("\n[5/6] Running MambaVSum model...")
    cfg = Config()
    cfg.device = device
    if device.type == "cpu":
        cfg.use_amp = False

    # Find checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try relative to script dir
        ckpt_path = SCRIPT_DIR / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"  Available checkpoints in {SCRIPT_DIR / 'checkpoints'}:")
        ckpt_dir = SCRIPT_DIR / "checkpoints"
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("*.pt")):
                print(f"    {f.name}")
        sys.exit(1)

    model = load_model(ckpt_path, cfg, device)

    # Map change points to subsampled space for the model
    # (the model expects change points in subsampled frame indices)
    cp_subsampled = change_points // 15  # approximate

    scores = predict_scores(model, visual_features, audio_features,
                            cp_subsampled, cfg)
    print(f"  Importance scores: min={scores.min():.3f}, "
          f"max={scores.max():.3f}, mean={scores.mean():.3f}")

    # ── Step 6: Generate summary & write video ────────────────────
    print("\n[6/6] Generating summary video...")

    summary_mask = generate_summary(
        scores, change_points, n_frames, n_frame_per_seg, picks,
        proportion=args.ratio
    )

    final_path = write_summary_video(
        str(video_path), str(output_path), summary_mask, fps
    )

    print("\n" + "=" * 60)
    print("  [OK] DONE! Summary video saved.")
    print(f"  Output: {final_path}")
    print("=" * 60)

    return final_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MambaVSum — Summarize any video using a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarize_video.py --video my_video.mp4
  python summarize_video.py --video my_video.mp4 --output summary.mp4
  python summarize_video.py --video my_video.mp4 --ratio 0.20
  python summarize_video.py --video my_video.mp4 --device cuda
        """,
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to the input video file (mp4, avi, mkv, etc.)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for the output summary video (default: <input>_summary.mp4)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/best_split4.pt",
        help="Path to the trained MambaVSum checkpoint (.pt file)",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.15,
        help="Summary ratio — fraction of original video to keep (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu)",
    )

    args = parser.parse_args()
    summarize(args)
