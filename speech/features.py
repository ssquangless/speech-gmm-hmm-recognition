"""Trích xuất đặc trưng MFCC 39 chiều (13 MFCC + delta + delta-delta) theo công thức HTK.

Chỉ dùng numpy/scipy (không phụ thuộc librosa) để mọi bước đều minh bạch và tái lập được.
Pipeline: tiền nhấn -> chia khung 25ms/10ms -> Hamming -> |FFT|^2 -> 26 bộ lọc mel -> log
-> DCT-II -> giữ 13 hệ số -> liftering -> delta/delta-delta -> CMVN theo từng utterance.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct

from config import Config, FeatureConfig
from speech.endpoint import find_speech_segment


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


_FB_CACHE: dict = {}


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Bộ lọc tam giác trên thang mel (HTK). Shape (n_mels, n_fft//2+1)."""
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (ce - lo)
        down = (hi - bins) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    _FB_CACHE[key] = fb
    return fb


def _deltas(feat: np.ndarray, window: int) -> np.ndarray:
    """Hệ số động theo hồi quy (HTK): d_t = sum_k k (c_{t+k}-c_{t-k}) / (2 sum k^2)."""
    T = len(feat)
    pad = np.pad(feat, ((window, window), (0, 0)), mode="edge")
    denom = 2.0 * sum(k * k for k in range(1, window + 1))
    out = np.zeros_like(feat)
    for k in range(1, window + 1):
        out += k * (pad[window + k:window + k + T] - pad[window - k:window - k + T])
    return out / denom


def mfcc_frames(y: np.ndarray, sr: int, fc: FeatureConfig) -> Optional[np.ndarray]:
    """MFCC (đã liftering) chưa có delta. Shape (T, n_ceps). None nếu quá ngắn."""
    y = np.asarray(y, dtype=np.float64)
    flen = int(round(sr * fc.frame_ms / 1000.0))
    hop = int(round(sr * fc.hop_ms / 1000.0))
    if len(y) < flen + hop:
        return None
    # tiền nhấn
    y = np.concatenate([[y[0] * (1.0 - fc.preemphasis)], y[1:] - fc.preemphasis * y[:-1]])
    frames = sliding_window_view(y, flen)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)          # bỏ DC từng khung
    frames = frames * np.hamming(flen)
    spec = np.abs(np.fft.rfft(frames, n=fc.n_fft, axis=1)) ** 2
    fb = mel_filterbank(sr, fc.n_fft, fc.n_mels, fc.fmin, min(fc.fmax, sr / 2.0))
    logmel = np.log(np.maximum(spec @ fb.T, fc.log_floor))
    ceps = dct(logmel, type=2, axis=1, norm="ortho")[:, :fc.n_ceps]
    if fc.lifter and fc.lifter > 0:
        n = np.arange(fc.n_ceps)
        ceps = ceps * (1.0 + (fc.lifter / 2.0) * np.sin(np.pi * n / fc.lifter))
    return ceps


def normalize(feat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return feat
    out = feat - feat.mean(axis=0, keepdims=True)
    if mode == "cmvn":
        out = out / (feat.std(axis=0, keepdims=True) + 1e-6)
    return out


def compute_features(y: np.ndarray, sr: int, fc: FeatureConfig) -> Optional[np.ndarray]:
    """MFCC + delta + delta-delta + chuẩn hóa. Shape (T, 39) khi dùng đủ cấu hình mặc định."""
    ceps = mfcc_frames(y, sr, fc)
    if ceps is None:
        return None
    parts = [ceps]
    d1 = _deltas(ceps, fc.delta_window)
    if fc.use_delta:
        parts.append(d1)
    if fc.use_delta2:
        parts.append(_deltas(d1, fc.delta_window))
    feat = np.concatenate(parts, axis=1)
    return normalize(feat, fc.norm)


def utterance_features(y: np.ndarray, cfg: Config, endpoint: bool = True,
                       return_segment: bool = False):
    """ĐIỂM VÀO DUY NHẤT: waveform float [-1,1] -> đặc trưng (T, D) hoặc None (không có tiếng nói).

    Được gọi bởi huấn luyện, đánh giá và demo, nên tiền xử lý không thể bị lệch nhau.
    """
    sr = cfg.audio.sample_rate
    y = np.asarray(y, dtype=np.float64)
    seg = (0, len(y))
    if endpoint:
        seg = find_speech_segment(y, sr, cfg.endpoint)
        if seg is None:
            return (None, None) if return_segment else None
    feat = compute_features(y[seg[0]:seg[1]], sr, cfg.features)
    if return_segment:
        return feat, seg
    return feat
