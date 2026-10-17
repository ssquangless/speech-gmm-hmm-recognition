"""Chỉ số đánh giá và các hàm trợ giúp (chỉ giữ những gì notebook và demo cần)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def score_matrix(models: Dict[str, object], seqs: Sequence[np.ndarray]) -> np.ndarray:
    """Tổng log-likelihood, shape (n_câu, n_từ) theo thứ tự models."""
    return np.array([[m.score(np.asarray(s, np.float64)) for m in models.values()] for s in seqs])


def wilson_interval(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def roc_points(pos: np.ndarray, neg: np.ndarray):
    """(ngưỡng, FAR, FRR) khi quét ngưỡng; chấp nhận nếu điểm >= ngưỡng."""
    thr = np.sort(np.unique(np.concatenate([pos, neg])))
    return thr, np.array([np.mean(neg >= t) for t in thr]), np.array([np.mean(pos < t) for t in thr])


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def choose_threshold(pos: np.ndarray, neg_noise: np.ndarray, target_frr: float = 0.03,
                     noise_far: float = 0.01, max_frr: float = 0.10) -> float:
    """Ngưỡng LLR/khung ưu tiên KHÔNG bỏ sót từ đúng: chỉ từ chối ~target_frr từ đúng; có thể nâng ngưỡng để
    nhiễu nền lọt qua <= noise_far nhưng không để bỏ sót vượt max_frr."""
    t_frr = float(np.quantile(pos, target_frr))
    t_noise = float(np.quantile(neg_noise, 1.0 - noise_far)) if len(neg_noise) else -np.inf
    return max(t_frr, min(t_noise, float(np.quantile(pos, max_frr))))


def fit_temperature(S: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature scaling: tìm tau để softmax(S / tau) có NLL nhỏ nhất trên validation.
    (Tổng LL của HMM bị 'thổi phồng' vì giả định các khung độc lập nên softmax thô luôn ~1.)"""
    ok = np.isfinite(S).all(axis=1)
    S, y = S[ok], np.asarray(y_idx)[ok]
    best_tau, best_nll = 1.0, np.inf
    for tau in np.logspace(0, 3, 121):
        Z = S / tau
        Z = Z - Z.max(axis=1, keepdims=True)
        nll = -(Z - np.log(np.exp(Z).sum(axis=1, keepdims=True)))[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_tau, best_nll = float(tau), float(nll)
    return best_tau


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_s, p_n = np.mean(clean.astype(np.float64) ** 2) + 1e-12, np.mean(noise.astype(np.float64) ** 2) + 1e-12
    return (clean + np.sqrt(p_s / (p_n * 10 ** (snr_db / 10.0))) * noise).astype(np.float32)


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    return (y * 10 ** (gain_db / 20.0)).astype(np.float32)


def rms_dbfs(y: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(y.astype(np.float64) ** 2) + 1e-12))
