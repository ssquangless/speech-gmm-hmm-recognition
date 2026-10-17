"""Cắt đoạn chứa từ nói (endpoint detection) bằng năng lượng khung.

Hàm ``find_speech_segment`` được dùng GIỐNG HỆT ở huấn luyện, đánh giá và demo trực tiếp,
nên đặc trưng đưa vào HMM luôn có cùng phân bố (từ nằm gọn trong đoạn, lề im lặng ngắn).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

from config import EndpointConfig


def frame_energy_db(y: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    """Năng lượng (dB) của từng khung 25 ms / bước 10 ms (đã trừ thành phần DC của khung)."""
    flen = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    if len(y) < flen:
        return np.zeros(0)
    frames = sliding_window_view(y.astype(np.float64), flen)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    e = np.mean(frames ** 2, axis=1)
    return 10.0 * np.log10(e + 1e-10)


def _runs(mask: np.ndarray):
    """Danh sách (start, end_inclusive) của các đoạn True liên tiếp."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def find_speech_segment(y: np.ndarray, sr: int, cfg: EndpointConfig) -> Optional[Tuple[int, int]]:
    """Trả về (start_sample, end_sample) của đoạn tiếng nói, hoặc None nếu không có tiếng nói."""
    e_db = frame_energy_db(y, sr, cfg.frame_ms, cfg.hop_ms)
    if len(e_db) < 5:
        return None
    hop = int(round(sr * cfg.hop_ms / 1000.0))
    flen = int(round(sr * cfg.frame_ms / 1000.0))

    sm = median_filter(e_db, size=3, mode="nearest")
    noise = np.percentile(sm, cfg.noise_percentile)
    peak = np.percentile(sm, cfg.peak_percentile)
    if peak - noise < cfg.min_dynamic_db:
        return None

    thr = max(noise + cfg.snr_floor_db, peak - cfg.peak_range_db)
    active = sm > thr

    min_run = max(1, int(round(cfg.min_run_ms / cfg.hop_ms)))
    max_gap = int(round(cfg.max_gap_ms / cfg.hop_ms))
    runs = [r for r in _runs(active) if (r[1] - r[0] + 1) >= min_run]
    if not runs:
        return None

    # gộp các đoạn cách nhau <= max_gap
    clusters = [[runs[0][0], runs[0][1]]]
    for s, e in runs[1:]:
        if s - clusters[-1][1] - 1 <= max_gap:
            clusters[-1][1] = e
        else:
            clusters.append([s, e])

    # chọn cụm có tổng năng lượng tuyến tính lớn nhất (bỏ qua tiếng động đơn lẻ)
    lin = 10.0 ** (e_db / 10.0)
    best = max(clusters, key=lambda c: lin[c[0]:c[1] + 1].sum())

    pad = int(round(sr * cfg.pad_ms / 1000.0))
    start = max(0, best[0] * hop - pad)
    end = min(len(y), best[1] * hop + flen + pad)
    if (end - start) * 1000.0 / sr < cfg.min_duration_ms:
        return None
    return int(start), int(end)
