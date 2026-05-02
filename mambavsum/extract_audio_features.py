# extract_audio_features.py
"""
Audio Feature Extraction for Multimodal Video Summarization.

Extracts audio features from video files using either:
  1. VGGish (Google's audio classification model) -> 128-d embeddings
  2. Mel-spectrogram + simple CNN -> 128-d embeddings (lightweight fallback)

Audio features are aligned with the visual frame subsampling (picks)
and saved into the CLIP h5 file under a separate dataset key.

Usage:
    python extract_audio_features.py \
        --source_h5  ../data/tvsum_clip_features.h5 \
        --video_dir  ./videos/tvsum/ \
        --output_h5  ../data/tvsum_clip_features.h5   # updates in-place

Requirements:
    pip install torch torchaudio h5py numpy tqdm
"""
import os
import argparse
import numpy as np
import h5py
import torch
import math
import pandas as pd
from pathlib import Path


def extract_audio_from_video(video_path, sr=16000):
    """
    Extract audio waveform from a video file.

    Returns:
        waveform: np.ndarray (n_samples,) mono, 16kHz
        sample_rate: int
    """
    try:
        import torchaudio
        # Try loading directly (works if torchaudio has a suitable backend)
        waveform, orig_sr = torchaudio.load(str(video_path))
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to target rate
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy(), sr
    except Exception as e:
        pass  # Fall through to ffmpeg approach

    # Fallback: use subprocess + ffmpeg binary (from imageio-ffmpeg or system)
    import subprocess
    import tempfile

    # Find ffmpeg binary: prefer imageio-ffmpeg (bundled), else system PATH
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = "ffmpeg"  # Hope it's on PATH

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                ffmpeg_exe, "-i", str(video_path),
                "-ac", "1", "-ar", str(sr),
                "-f", "wav", "-y", tmp_path,
            ],
            capture_output=True, check=True,
        )
        import wave
        with wave.open(tmp_path, "rb") as wf:
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            samples /= 32768.0  # Normalize to [-1, 1]
        return samples, sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_mel_spectrogram(waveform, sr=16000, n_mels=64, hop_length=160, n_fft=400):
    """
    Compute log-mel spectrogram from waveform.

    Returns:
        mel_spec: np.ndarray (n_time_steps, n_mels)
    """
    try:
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        waveform_t = torch.from_numpy(waveform).unsqueeze(0)
        mel = mel_transform(waveform_t)  # (1, n_mels, n_time)
        mel = torch.log(mel + 1e-8)
        return mel.squeeze(0).T.numpy()  # (n_time, n_mels)

    except ImportError:
        # Pure numpy fallback using STFT
        n_frames_audio = 1 + (len(waveform) - n_fft) // hop_length
        mel_spec = np.zeros((n_frames_audio, n_mels), dtype=np.float32)

        window = np.hanning(n_fft).astype(np.float32)

        for i in range(n_frames_audio):
            start = i * hop_length
            frame = waveform[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed)) ** 2

            # Simple mel filterbank (linear approximation)
            freq_bins = len(spectrum)
            mel_points = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            bin_points = (hz_points * n_fft / sr).astype(int)
            bin_points = np.clip(bin_points, 0, freq_bins - 1)

            for m in range(n_mels):
                start_bin = bin_points[m]
                peak_bin = bin_points[m + 1]
                end_bin = bin_points[m + 2]
                if peak_bin > start_bin:
                    mel_spec[i, m] = np.mean(spectrum[start_bin:end_bin + 1])

            mel_spec[i] = np.log(mel_spec[i] + 1e-8)

        return mel_spec


class SimpleAudioEncoder(torch.nn.Module):
    """
    Lightweight CNN encoder: mel-spectrogram -> 128-d audio embeddings.

    Takes (batch, n_mels, time_window) and outputs (batch, 128).
    Used when VGGish is not available.
    """
    def __init__(self, n_mels=64, out_dim=128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.fc = torch.nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (batch, n_mels, time)
        h = self.encoder(x).squeeze(-1)  # (batch, 128)
        return self.fc(h)  # (batch, out_dim)


def extract_audio_features_for_video(
    waveform, sr, picks, n_frames_total, fps, audio_dim=128, n_mels=64
):
    """
    Extract audio features aligned with visual frame subsampling.

    For each subsampled frame index in `picks`, extract a window of audio
    centered at the corresponding timestamp and compute mel features.

    Returns:
        audio_features: np.ndarray (n_picks, audio_dim)
    """
    # Compute mel spectrogram for entire audio
    mel_spec = compute_mel_spectrogram(waveform, sr, n_mels=n_mels)
    total_mel_frames = mel_spec.shape[0]

    # Compute audio-to-video time mapping
    audio_fps = sr / 160  # mel hop_length = 160 at sr=16000 => 100 fps

    features = np.zeros((len(picks), audio_dim), dtype=np.float32)

    # Window size in mel frames (covers ~0.5 seconds centered on each pick)
    window_size = int(audio_fps * 0.5)

    for i, pick_idx in enumerate(picks):
        # Convert video frame index to time in seconds
        time_sec = pick_idx / max(fps, 1)

        # Convert time to mel frame index
        mel_center = int(time_sec * audio_fps)
        mel_start = max(0, mel_center - window_size // 2)
        mel_end = min(total_mel_frames, mel_center + window_size // 2)

        if mel_start < mel_end and mel_start < total_mel_frames:
            segment = mel_spec[mel_start:mel_end]  # (window, n_mels)
            # Pool: mean + std concatenation, then truncate/pad to audio_dim
            mean_feat = segment.mean(axis=0)  # (n_mels,)
            std_feat = segment.std(axis=0)    # (n_mels,)
            combined = np.concatenate([mean_feat, std_feat])  # (2 * n_mels,)

            # Truncate or pad to audio_dim
            if len(combined) >= audio_dim:
                features[i] = combined[:audio_dim]
            else:
                features[i, :len(combined)] = combined

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-8)

    return features


# ---------------------------------------------------------------------------
# Filename resolution (same helpers as extract_clip_features.py)
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mkv", ".webm", ".mov"]


def find_video_by_name(video_dir, name):
    """Exact stem lookup first, then partial match."""
    for ext in VIDEO_EXTENSIONS:
        candidate = Path(video_dir) / (name + ext)
        if candidate.exists():
            return candidate
    for p in Path(video_dir).iterdir():
        if p.suffix.lower() in VIDEO_EXTENSIONS and name in p.stem:
            return p
    return None


def build_map_from_tsv(h5_keys, video_dir, info_tsv_path):
    """
    Build {h5_key: Path | None} using ydata-tvsum50-info.tsv.
    h5 keys (video_1 … video_50) sorted numerically, zipped with TSV rows.
    """
    info_tsv_path = Path(info_tsv_path)
    if not info_tsv_path.exists():
        raise FileNotFoundError(f"Info TSV not found: {info_tsv_path}")

    df = pd.read_csv(info_tsv_path, sep="\t", header=None)
    first_val = str(df.iloc[0, 1])
    if len(first_val) != 11:  # Likely a header row
        df = pd.read_csv(info_tsv_path, sep="\t", header=0)

    youtube_ids = df.iloc[:, 1].astype(str).str.strip().tolist()

    def numeric_key(k):
        try:
            return int(k.split("_")[-1])
        except ValueError:
            return 0

    sorted_keys = sorted(h5_keys, key=numeric_key)

    print(f"\n  {'H5 Key':<12} {'YouTube ID':<14} {'Status'}")
    print("  " + "-" * 50)

    mapping = {}
    for key, yt_id in zip(sorted_keys, youtube_ids):
        video_path = find_video_by_name(video_dir, yt_id)
        mapping[key] = video_path
        status = f"found  {video_path.name}" if video_path else "MISSING"
        print(f"  {key:<12} {yt_id:<14} {status}")

    found = sum(1 for v in mapping.values() if v is not None)
    print(f"\n  Mapped: {found} / {len(mapping)} videos found on disk.\n")
    return mapping


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_audio_for_dataset(
    source_h5_path,
    video_dir,
    output_h5_path,
    info_tsv="./ydata-tvsum50-info.tsv",
    audio_dim=128,
):
    """
    Extract real audio features from video files and save into the h5 file.

    For each video:
      1. Extract audio waveform (torchaudio or ffmpeg fallback)
      2. Compute mel-spectrogram
      3. Align audio windows with the visual frame picks
      4. Produce 128-d features per picked frame
      5. Store as 'audio_features' dataset in the h5 file
    """
    from tqdm import tqdm
    import cv2

    same_file = os.path.abspath(source_h5_path) == os.path.abspath(output_h5_path)

    print("=" * 60)
    print("  EXTRACTING REAL AUDIO FEATURES FROM VIDEOS")
    print("=" * 60)
    print(f"  Source h5  : {source_h5_path}")
    print(f"  Video dir  : {video_dir}")
    print(f"  Output h5  : {output_h5_path}")
    print(f"  In-place   : {same_file}")
    print(f"  Audio dim  : {audio_dim}")

    # --- Read h5 keys and build video mapping ---
    with h5py.File(source_h5_path, "r") as src:
        h5_keys = sorted(list(src.keys()),
                         key=lambda k: int(k.split('_')[-1]) if k.split('_')[-1].isdigit() else 0)

    key_to_path = build_map_from_tsv(h5_keys, video_dir, info_tsv)

    # --- Process each video ---
    success_count = 0
    fail_count = 0

    # Open h5 in append mode if in-place, else copy everything first
    if same_file:
        h5_file = h5py.File(output_h5_path, "a")
    else:
        # Copy entire source then add audio features
        import shutil
        shutil.copy2(source_h5_path, output_h5_path)
        h5_file = h5py.File(output_h5_path, "a")

    try:
        for key in tqdm(h5_keys, desc="Extracting audio features"):
            video_path = key_to_path.get(key)
            picks = np.array(h5_file[key]["picks"][()])
            n_frames = int(h5_file[key]["n_frames"][()])

            if video_path is None or not Path(video_path).exists():
                tqdm.write(f"  {key}: video not found — using zeros")
                audio_feat = np.zeros((len(picks), audio_dim), dtype=np.float32)
                fail_count += 1
            else:
                try:
                    # Get FPS from video
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0
                    cap.release()

                    # Extract audio waveform
                    waveform, sr = extract_audio_from_video(video_path)
                    tqdm.write(
                        f"  {key}: {Path(video_path).name}  "
                        f"audio={len(waveform)/sr:.1f}s  fps={fps:.1f}  "
                        f"picks={len(picks)}"
                    )

                    # Compute aligned audio features
                    audio_feat = extract_audio_features_for_video(
                        waveform, sr, picks, n_frames, fps,
                        audio_dim=audio_dim,
                    )
                    success_count += 1

                except Exception as e:
                    tqdm.write(f"  {key}: ERROR {e} — using zeros")
                    audio_feat = np.zeros((len(picks), audio_dim), dtype=np.float32)
                    fail_count += 1

            # Save audio features
            if "audio_features" in h5_file[key]:
                del h5_file[key]["audio_features"]
            h5_file[key].create_dataset("audio_features", data=audio_feat)

    finally:
        h5_file.close()

    print(f"\n{'=' * 60}")
    print(f"  [Done] Audio features saved to: {output_h5_path}")
    print(f"  Succeeded : {success_count} / {len(h5_keys)}")
    print(f"  Failed    : {fail_count} / {len(h5_keys)}")
    print(f"  Feature   : {audio_dim}-d per picked frame")
    print(f"{'=' * 60}")


def generate_synthetic_audio_features(source_h5_path, output_h5_path, audio_dim=128):
    """
    Generate synthetic audio features for testing.
    Creates random but consistent 128-d features for each video.
    """
    print("=" * 60)
    print("  GENERATING SYNTHETIC AUDIO FEATURES (for testing only)")
    print("=" * 60)

    rng = np.random.default_rng(99)

    with h5py.File(source_h5_path, "r") as src:
        keys = sorted(list(src.keys()))
        # Check if output is same file
        same_file = os.path.abspath(source_h5_path) == os.path.abspath(output_h5_path)

    mode = "a" if same_file else "w"

    with h5py.File(output_h5_path, mode) as dst:
        for key in keys:
            if same_file:
                n_frames_sub = dst[key]["features"].shape[0]
            else:
                with h5py.File(source_h5_path, "r") as src:
                    n_frames_sub = src[key]["features"].shape[0]

            # Generate random audio features
            audio_feat = rng.standard_normal((n_frames_sub, audio_dim)).astype(np.float32)
            norms = np.linalg.norm(audio_feat, axis=1, keepdims=True)
            audio_feat = audio_feat / (norms + 1e-8)

            # Save under "audio_features" key
            if same_file and "audio_features" in dst[key]:
                del dst[key]["audio_features"]
            if same_file:
                dst[key].create_dataset("audio_features", data=audio_feat)
            else:
                if key not in dst:
                    grp = dst.create_group(key)
                    # Copy everything from source
                    with h5py.File(source_h5_path, "r") as src:
                        for k in src[key].keys():
                            grp.create_dataset(k, data=src[key][k][()])
                dst[key].create_dataset("audio_features", data=audio_feat)

    print(f"[OK] Synthetic audio features ({audio_dim}-d) saved to: {output_h5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio features for multimodal video summarization"
    )
    parser.add_argument(
        "--source_h5", type=str,
        default="../data/tvsum_clip_features.h5",
        help="H5 file to add audio features to",
    )
    parser.add_argument(
        "--video_dir", type=str,
        default="./videos/tvsum/",
        help="Directory containing raw video files",
    )
    parser.add_argument(
        "--output_h5", type=str,
        default="../data/tvsum_clip_features.h5",
        help="Output h5 file (can be same as source to update in-place)",
    )
    parser.add_argument(
        "--info_tsv", type=str,
        default="./ydata-tvsum50-info.tsv",
        help="Path to ydata-tvsum50-info.tsv for h5-key to video-file mapping.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic audio features for testing",
    )

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_audio_features(args.source_h5, args.output_h5)
    else:
        extract_audio_for_dataset(
            source_h5_path=args.source_h5,
            video_dir=args.video_dir,
            output_h5_path=args.output_h5,
            info_tsv=args.info_tsv,
        )
