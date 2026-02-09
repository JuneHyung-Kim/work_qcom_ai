import os
import tarfile
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_SR = 16000
N_MELS = 80

CLIP_SECONDS = 5.0
FRAMES_PER_SEC = 100  # hop=160 @16k
TARGET_FRAMES = int(CLIP_SECONDS * FRAMES_PER_SEC)  # 500

LIBRISPEECH_TGZ = Path("dev-clean.tar.gz")   # 필요 시 경로 수정
FLEURS_KO_DIR   = Path("fleurs_ko_subset")

WORK_DIR = Path("./_work_calib")
LIBRI_EXTRACT_DIR = WORK_DIR / "librispeech_dev_clean"
OUT_DIR = Path("./calib_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

N_EN = 200
N_KO = 200


# -----------------------------
# Whisper-like log-mel (torchaudio)
# -----------------------------
_mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=400,          # 25ms @16k
    win_length=400,
    hop_length=160,     # 10ms @16k
    n_mels=N_MELS,
    f_min=0,
    f_max=8000,
    power=2.0,
    normalized=False,
)

def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    # (C, T)
    if wav.ndim != 2:
        raise ValueError(f"wav rank must be 2, got {wav.shape}")
    if wav.shape[0] == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)

def _resample_if_needed(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return wav
    return torchaudio.functional.resample(wav, sr, TARGET_SR)

def _random_crop_or_pad(wav: torch.Tensor, sr: int, seconds: float) -> torch.Tensor:
    target_len = int(sr * seconds)
    n = wav.shape[-1]
    if n == target_len:
        return wav
    if n > target_len:
        start = random.randint(0, n - target_len)
        return wav[..., start:start + target_len]
    pad_len = target_len - n
    return torch.nn.functional.pad(wav, (0, pad_len))

def audio_to_logmel_80xT(wav: torch.Tensor, sr: int) -> np.ndarray:
    wav = _to_mono(wav)
    wav = _resample_if_needed(wav, sr)

    mel = _mel_spec(wav)  # (1,80,T)
    mel = mel.clamp(min=1e-10).log10()
    mel = mel.squeeze(0)  # (80,T)
    return mel.cpu().numpy().astype(np.float32)

def fix_frames(mel_80xT: np.ndarray, target_frames: int) -> np.ndarray:
    T = mel_80xT.shape[1]
    if T > target_frames:
        return mel_80xT[:, :target_frames]
    if T < target_frames:
        return np.pad(mel_80xT, ((0,0), (0, target_frames - T)), mode="constant")
    return mel_80xT


# -----------------------------
# Robust audio loader
# -----------------------------
def load_audio_any(path: Path):
    """
    Returns wav (torch.Tensor shape (1,T), float32), sr (int)
    Uses soundfile first (good for FLAC/WAV), then torchaudio as fallback.
    """
    # 1) soundfile (recommended on Windows for FLAC)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
        data = data.T  # (C, T)
        wav = torch.from_numpy(data)
        return wav, sr
    except Exception:
        pass

    # 2) torchaudio fallback
    wav, sr = torchaudio.load(str(path))
    return wav, sr


# -----------------------------
# Dataset discovery
# -----------------------------
def extract_librispeech_if_needed(tgz_path: Path, out_dir: Path):
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {tgz_path} -> {out_dir}")
    # Python 3.14 tar safety 변화 대비: filter='data' 사용
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(out_dir, filter="data")
    marker.write_text("ok")

def find_audio_files(root: Path, exts=(".wav", ".flac")):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)

def build_mels_from_files(files, n_pick: int, tag: str):
    files = list(files)
    random.shuffle(files)
    picks = files[: min(n_pick, len(files))]

    mels = []
    n_ok = 0
    n_fail = 0

    for p in tqdm(picks, desc=f"mel:{tag}"):
        try:
            wav, sr = load_audio_any(p)
            wav = _random_crop_or_pad(wav, sr, CLIP_SECONDS)
            mel = audio_to_logmel_80xT(wav, sr)   # (80,T)
            mel = fix_frames(mel, TARGET_FRAMES)  # (80,500)
            mel = mel[None, :, :]                 # (1,80,500)
            mels.append(mel)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            continue

    print(f"[{tag}] ok={n_ok}, fail={n_fail}")

    if not mels:
        raise RuntimeError(f"No usable samples for {tag} (all failed to decode?)")

    arr = np.stack(mels, axis=0).astype(np.float32)  # (N,1,80,500)
    return arr


def main():
    if not LIBRISPEECH_TGZ.exists():
        raise FileNotFoundError(f"Missing {LIBRISPEECH_TGZ.resolve()}")

    extract_librispeech_if_needed(LIBRISPEECH_TGZ, LIBRI_EXTRACT_DIR)

    libri_root = LIBRI_EXTRACT_DIR
    if (LIBRI_EXTRACT_DIR / "LibriSpeech").exists():
        libri_root = LIBRI_EXTRACT_DIR / "LibriSpeech"

    en_files = find_audio_files(libri_root, exts=(".flac", ".wav"))
    print(f"[found] EN files: {len(en_files)} in {libri_root}")

    if not FLEURS_KO_DIR.exists():
        raise FileNotFoundError(f"Missing {FLEURS_KO_DIR.resolve()}")
    ko_files = find_audio_files(FLEURS_KO_DIR, exts=(".wav", ".flac"))
    print(f"[found] KO files: {len(ko_files)} in {FLEURS_KO_DIR}")

    en_mels = build_mels_from_files(en_files, N_EN, "en")
    ko_mels = build_mels_from_files(ko_files, N_KO, "ko")

    mels = np.concatenate([en_mels, ko_mels], axis=0)  # (N,1,80,500)

    out_path = OUT_DIR / "calib_whisper_5s_enko.npz"
    np.savez_compressed(out_path, mels=mels)

    print(f"[ok] saved: {out_path}")
    print(f"[shape] mels: {mels.shape} dtype={mels.dtype}")


if __name__ == "__main__":
    main()
