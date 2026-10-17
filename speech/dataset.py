"""Nạp Google Speech Commands v0.02, chia train/val/test THEO NGƯỜI NÓI, trích đặc trưng có cache.

Thông tin về dữ liệu gốc (Warden, 2018, arXiv:1804.03209):
  * 105,829 clip, 35 từ, 2,618 người nói; WAV 16 kHz, 16-bit, mono, dài <= 1 giây.
  * Thu bằng trình duyệt qua mic điện thoại/laptop (chất lượng thấp, có nhiễu) rồi nén OGG -> WAV.
  * Mỗi clip là đoạn 1 giây "to nhất" cắt từ bản thu 1.5 giây (từ nằm gần giữa clip);
    clip có độ lớn trung bình |x| < 0.004 đã bị loại.
  * Tên file: ``<speaker_id>_nohash_<k>.wav``; speaker_id là mã băm 8 ký tự hex.
  * ``validation_list.txt`` / ``testing_list.txt`` do hàm băm trên tên người nói quyết định,
    nên một người chỉ nằm trong MỘT tập. Hàm ``verify_speaker_disjoint`` kiểm tra lại điều này.
  * Thư mục ``_background_noise_`` có 6 file nhiễu dài ~1 phút.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

try:                                    # soundfile nhanh hơn; thiếu thì dùng scipy
    import soundfile as sf
except ImportError:                     # pragma: no cover
    sf = None

from config import Config
from speech.features import utterance_features, compute_features


# ----------------------------------------------------------------------------- tải / liệt kê
def download_speech_commands(root: str, url: str) -> None:
    """Tải và giải nén (khoảng 2.3 GB tải, 3.8 GB sau giải nén). Bỏ qua nếu đã có."""
    root = Path(root)
    if (root / "testing_list.txt").exists():
        print("Đã có dữ liệu tại", root)
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print("Đang tải", url)
        urllib.request.urlretrieve(url, archive)
    print("Đang giải nén...")
    with tarfile.open(archive) as tf:
        tf.extractall(root)
    print("Xong.")


def speaker_of(filename: str) -> str:
    return os.path.basename(filename).split("_nohash_")[0]


def _read_list(path: Path) -> set:
    if not path.exists():
        return set()
    return {ln.strip().replace("\\", "/") for ln in path.read_text().splitlines() if ln.strip()}


def list_utterances(root: str, words: List[str]) -> List[dict]:
    """Danh sách {path, word, speaker, split} cho các từ yêu cầu, theo danh sách chính thức."""
    root = Path(root)
    val, test = _read_list(root / "validation_list.txt"), _read_list(root / "testing_list.txt")
    items = []
    for w in words:
        d = root / w
        if not d.is_dir():
            raise FileNotFoundError(f"Không thấy thư mục từ '{w}' trong {root}")
        for f in sorted(d.glob("*.wav")):
            rel = f"{w}/{f.name}"
            split = "test" if rel in test else "val" if rel in val else "train"
            items.append(dict(path=str(f), word=w, speaker=speaker_of(f.name), split=split))
    return items


def other_words(root: str, targets: List[str]) -> List[str]:
    root = Path(root)
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith("_") and d.name not in targets)


def verify_speaker_disjoint(items: List[dict]) -> Dict[str, int]:
    """Trả về số người nói xuất hiện ở nhiều hơn một tập (phải = 0)."""
    sp = {s: {i["speaker"] for i in items if i["split"] == s} for s in ("train", "val", "test")}
    return {"train&val": len(sp["train"] & sp["val"]), "train&test": len(sp["train"] & sp["test"]),
            "val&test": len(sp["val"] & sp["test"])}


def resplit_by_speaker(items: List[dict], frac=(0.8, 0.1, 0.1)) -> List[dict]:
    """Dự phòng: tự chia theo người nói bằng hàm băm ổn định (khi danh sách chính thức bị thiếu/lệch)."""
    out = []
    for it in items:
        h = int(hashlib.sha1(it["speaker"].encode()).hexdigest(), 16) % 1000 / 1000.0
        split = "train" if h < frac[0] else "val" if h < frac[0] + frac[1] else "test"
        out.append({**it, "split": split})
    return out


def get_items(cfg: Config, words: List[str]) -> List[dict]:
    items = list_utterances(cfg.data.data_root, words)
    ov = verify_speaker_disjoint(items)
    if any(ov.values()) or not any(i["split"] != "train" for i in items):
        warnings.warn(f"Danh sách chính thức không tách người nói ({ov}); chia lại theo băm người nói.")
        items = resplit_by_speaker(items)
        ov = verify_speaker_disjoint(items)
    assert not any(ov.values()), f"Vẫn còn người nói trùng giữa các tập: {ov}"
    return items


# ----------------------------------------------------------------------------- đọc wav / nhiễu
def read_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Đọc WAV thành float32 trong [-1, 1] (mono)."""
    if sf is not None:
        y, fs = sf.read(path, dtype="float32", always_2d=False)
    else:
        from scipy.io import wavfile
        fs, y = wavfile.read(path)
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        else:
            y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if fs != sr:
        raise ValueError(f"{path}: tần số lấy mẫu {fs} != {sr}")
    return y


def load_noise_bank(root: str, sr: int = 16000) -> Dict[str, np.ndarray]:
    nd = Path(root) / "_background_noise_"
    return {f.stem: read_wav(str(f), sr) for f in sorted(nd.glob("*.wav"))}


def noise_part(y: np.ndarray, part: str) -> np.ndarray:
    """70% đầu của mỗi file nhiễu dùng để huấn luyện mô hình nền; 30% cuối dùng để đánh giá/trộn nhiễu."""
    cut = int(0.7 * len(y))
    return y[:cut] if part == "train" else y[cut:]


def sample_noise_clips(bank: Dict[str, np.ndarray], part: str, n: int, seconds: float,
                       sr: int, rng: np.random.Generator) -> List[np.ndarray]:
    names = list(bank)
    L = int(seconds * sr)
    out = []
    for _ in range(n):
        y = noise_part(bank[names[int(rng.integers(len(names)))]], part)
        s = int(rng.integers(0, max(1, len(y) - L)))
        out.append(y[s:s + L].copy())
    return out


# ----------------------------------------------------------------------------- đặc trưng + cache
def _feat_one(path: str, cfg: Config):
    y = read_wav(path, cfg.audio.sample_rate)
    f = utterance_features(y, cfg)
    return None if f is None else f.astype(np.float32)


def extract_many(paths: List[str], cfg: Config, n_jobs: int = -1):
    res = joblib.Parallel(n_jobs=n_jobs, batch_size=64)(joblib.delayed(_feat_one)(p, cfg) for p in paths)
    return res


def build_feature_sets(cfg: Config, n_jobs: int = -1, use_cache: bool = True, verbose: bool = True) -> dict:
    """Đặc trưng cho: 4 từ đích (train/val/test) + 'unknown' (từ khác) + nhiễu nền.

    Trả về dict:
      sets[split][word] -> list[np.ndarray]    (chỉ gồm các clip còn lại sau endpoint)
      meta[split][word] -> list[dict(path, speaker)]
      stats -> số clip bị loại bởi endpoint detector theo (split, word)
      unknown -> {"train": [...], "val": [...], "test": [...]} đặc trưng của các từ khác (nhãn "unknown")
      noise -> {"train": [...], "val": [...]}  đặc trưng nhiễu nền (CMVN, không endpoint)
    """
    from config import TARGET_WORDS
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{cfg.frontend_hash()}_{cfg.data.filler_per_word}_{cfg.data.noise_segments}"
    cache_file = cache_dir / f"features_{key}.joblib"
    if use_cache and cache_file.exists():
        if verbose:
            print("Nạp cache đặc trưng:", cache_file)
        return joblib.load(cache_file)

    words = TARGET_WORDS
    items = get_items(cfg, words)
    sets = {s: {w: [] for w in words} for s in ("train", "val", "test")}
    meta = {s: {w: [] for w in words} for s in ("train", "val", "test")}
    stats = {}
    for s in ("train", "val", "test"):
        for w in words:
            sel = [i for i in items if i["split"] == s and i["word"] == w]
            if verbose:
                print(f"Trích đặc trưng {s}/{w}: {len(sel)} clip", flush=True)
            feats = extract_many([i["path"] for i in sel], cfg, n_jobs)
            kept = [(f, i) for f, i in zip(feats, sel) if f is not None]
            sets[s][w] = [f for f, _ in kept]
            meta[s][w] = [dict(path=i["path"], speaker=i["speaker"]) for _, i in kept]
            stats[(s, w)] = dict(total=len(sel), kept=len(kept))

    # ---- "unknown": các từ khác, lấy tối đa filler_per_word clip/từ ở mỗi tập
    rng = np.random.default_rng(cfg.model.seed)
    unk_words = other_words(cfg.data.data_root, words)
    unk_items = get_items(cfg, unk_words)
    unknown = {"train": [], "val": [], "test": []}
    for s in unknown:
        sel = []
        for w in unk_words:
            c = [i for i in unk_items if i["split"] == s and i["word"] == w]
            k = cfg.data.filler_per_word if s == "train" else max(20, cfg.data.filler_per_word // 4)
            idx = rng.permutation(len(c))[:k]
            sel += [c[j] for j in idx]
        if verbose:
            print(f"Trích đặc trưng unknown/{s}: {len(sel)} clip", flush=True)
        unknown[s] = [f for f in extract_many([i["path"] for i in sel], cfg, n_jobs) if f is not None]

    # ---- nhiễu nền
    sr = cfg.audio.sample_rate
    bank = load_noise_bank(cfg.data.data_root, sr)
    noise = {}
    for part, n in (("train", cfg.data.noise_segments), ("val", max(50, cfg.data.noise_segments // 5))):
        clips = sample_noise_clips(bank, "train" if part == "train" else "eval", n, 0.6, sr, rng)
        noise[part] = [f.astype(np.float32) for f in
                       (compute_features(c, sr, cfg.features) for c in clips) if f is not None]

    out = dict(sets=sets, meta=meta, stats=stats, unknown=unknown, noise=noise)
    if use_cache:
        joblib.dump(out, cache_file)
    return out
