"""HMM-GMM cho từng từ bằng hmmlearn.GMMHMM, kèm lưu/nạp mô hình.

Các thiết lập quan trọng được giữ (xem tools/check_hmmlearn.py để kiểm tra trên máy bạn):
  1. Left-to-right: ``transmat_`` có số 0 -> hmmlearn giữ nguyên số 0 khi EM; ``startprob_`` = [1,0,...,0].
  2. Khởi tạo bằng PHÂN ĐOẠN ĐỀU (rồi bằng Viterbi), không dùng k-means ngẫu nhiên:
     ``init_params=""`` để hmmlearn không ghi đè tham số ta đặt.
  3. Huấn luyện lại: phân đoạn lại bằng Viterbi (``predict``), khởi tạo lại, chỉ nhận nếu val LL tốt hơn.
  4. Gaussian đường chéo, sàn phương sai ``min_covar``, dùng toàn bộ dữ liệu, early stopping theo validation.

Giới hạn so với bản tự cài đặt trước đây (hmmlearn không có): không bắt buộc kết thúc ở trạng thái cuối,
không chia dần mixture (thay bằng GaussianMixture trên từng trạng thái), không có MAP adaptation.
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from hmmlearn.hmm import GMMHMM
from sklearn.mixture import GaussianMixture

from config import Config, ModelConfig


# ----------------------------------------------------------------------------- khởi tạo
def uniform_paths(seqs: Sequence[np.ndarray], n_states: int) -> List[np.ndarray]:
    """Chia mỗi utterance thành n_states đoạn bằng nhau."""
    return [np.minimum((np.arange(len(x)) * n_states) // len(x), n_states - 1) for x in seqs]


def make_model(n_states: int, n_mix: int, mc: ModelConfig, seed: int, startprob, transmat,
               weights, means, covars) -> GMMHMM:
    m = GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type="diag", min_covar=mc.min_covar,
               n_iter=1, tol=0.0, init_params="", params="stmcw", random_state=seed)
    m.startprob_, m.transmat_ = startprob, transmat
    m.weights_, m.means_, m.covars_ = weights, means, covars
    return m


def init_from_paths(seqs: Sequence[np.ndarray], paths: Sequence[np.ndarray], n_states: int,
                    mc: ModelConfig, seed: int) -> GMMHMM:
    """Khởi tạo tham số từ một phân đoạn (đều hoặc Viterbi): mỗi trạng thái một GMM, ma trận chuyển L-R."""
    rng = np.random.default_rng(seed)
    X, P = np.concatenate(seqs).astype(np.float64), np.concatenate(paths)
    N, M, D = n_states, mc.n_mix, X.shape[1]
    w, mu, var = np.zeros((N, M)), np.zeros((N, M, D)), np.ones((N, M, D))
    for j in range(N):
        Xj = X[P == j]
        if len(Xj) >= 10 * M:
            g = GaussianMixture(M, covariance_type="diag", reg_covar=mc.min_covar,
                                max_iter=30, random_state=seed).fit(Xj)
            w[j], mu[j], var[j] = g.weights_, g.means_, g.covariances_
        else:                                           # quá ít khung: M Gaussian gần giống nhau
            m0 = Xj.mean(0) if len(Xj) else X.mean(0)
            v0 = np.maximum(Xj.var(0) if len(Xj) > 1 else X.var(0), mc.min_covar)
            w[j], var[j] = 1.0 / M, v0
            mu[j] = m0 + 0.1 * np.sqrt(v0) * rng.standard_normal((M, D))
    counts = np.zeros((N, N))
    for p in paths:
        np.add.at(counts, (p[:-1], p[1:]), 1.0)
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i], A[i, i + 1] = counts[i, i] + 0.5, counts[i, i + 1] + 0.5
    A[N - 1, N - 1] = 1.0
    A /= A.sum(axis=1, keepdims=True)
    start = np.zeros(N)
    start[0] = 1.0
    return make_model(N, M, mc, seed, start, A, w, mu, np.maximum(var, mc.min_covar))


# ----------------------------------------------------------------------------- huấn luyện
def _ll_per_frame(model: GMMHMM, seqs: Sequence[np.ndarray]) -> float:
    if not seqs:
        return float("nan")
    return float(model.score(np.concatenate(seqs), [len(s) for s in seqs]) / sum(len(s) for s in seqs))


def _train_ll(model: GMMHMM, X: np.ndarray, lengths: List[int]) -> float:
    try:
        return float(model.monitor_.history[-1])        # log-likelihood của vòng EM vừa chạy
    except Exception:
        return float(model.score(X, lengths))


def train_word(name: str, train_seqs: Sequence[np.ndarray], val_seqs: Sequence[np.ndarray],
               n_states: int, mc: ModelConfig, seed: int = 0, verbose: bool = False):
    """Trả về (model_tốt_nhất, history). history: list dict(word, round, iter, train_ll, val_ll)."""
    train_seqs = [np.asarray(s, np.float64) for s in train_seqs]
    val_seqs = [np.asarray(s, np.float64) for s in val_seqs]
    X, lengths = np.concatenate(train_seqs), [len(s) for s in train_seqs]
    paths = uniform_paths(train_seqs, n_states)
    best, best_val, hist = None, -np.inf, []
    for rnd in range(mc.realign_rounds + 1):
        model = init_from_paths(train_seqs, paths, n_states, mc, seed + rnd)
        cand, cand_val, bad = None, -np.inf, 0
        for _ in range(mc.n_iter):
            model.fit(X, lengths)                       # đúng 1 vòng EM (n_iter=1), tiếp tục từ tham số hiện tại
            tr, va = _train_ll(model, X, lengths) / len(X), _ll_per_frame(model, val_seqs)
            hist.append(dict(word=name, round=rnd, iter=len(hist) + 1, train_ll=tr, val_ll=va))
            if verbose:
                print(f"[{name}] round {rnd} it {len(hist):2d} train {tr:8.3f} | val {va:8.3f}", flush=True)
            if np.isnan(va) or va > cand_val + 1e-3:
                cand, cand_val, bad = copy.deepcopy(model), va, 0
            else:
                bad += 1
                if bad >= mc.patience:
                    break
        if best is not None and not cand_val > best_val + 1e-4:
            break                                       # khởi tạo lại không tốt hơn -> dừng
        best, best_val = cand, cand_val
        if rnd < mc.realign_rounds:
            states = best.predict(X, lengths)           # Viterbi
            paths = np.split(states, np.cumsum(lengths)[:-1])
    return best, hist


def train_all(train_by_word: Dict[str, list], val_by_word: Dict[str, list], cfg: Config,
              n_jobs: int = 4, verbose: bool = False):
    """Huấn luyện song song các mô hình từ. Trả về (models, history)."""
    mc = cfg.model
    jobs = [joblib.delayed(train_word)(w, train_by_word[w], val_by_word.get(w, []), mc.n_states[w], mc,
                                       mc.seed + 10 * i, verbose and n_jobs == 1)
            for i, w in enumerate(train_by_word)]
    res = joblib.Parallel(n_jobs=n_jobs)(jobs)
    return {w: r[0] for w, r in zip(train_by_word, res)}, [h for r in res for h in r[1]]


def train_filler(seqs: Sequence[np.ndarray], n_mix: int, seed: int = 0) -> GaussianMixture:
    """Mô hình nền (không có cấu trúc thời gian) trên khung của 'từ khác' + nhiễu, dùng để từ chối."""
    X = np.concatenate([np.asarray(s, np.float64) for s in seqs])
    return GaussianMixture(n_mix, covariance_type="diag", reg_covar=1e-2, max_iter=60, random_state=seed).fit(X)


def filler_loglik(filler: GaussianMixture, seqs: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([filler.score_samples(np.asarray(s, np.float64)).sum() for s in seqs])


# ----------------------------------------------------------------------------- lưu / nạp
def save_bundle(path, cfg: Config, models: dict, filler, thresholds: Optional[dict] = None,
                meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(dict(version=3, config=cfg.to_dict(), models=models, filler=filler,
                     thresholds=thresholds or dict(min_llr_per_frame=-np.inf, temperature=1.0),
                     meta=meta or {}), path)


def load_bundle(path) -> dict:
    b = joblib.load(path)
    if b.get("version") != 3:
        raise RuntimeError(f"{path} là file mô hình của phiên bản cũ. Hãy chạy lại notebooks/02_train.ipynb.")
    return dict(config=Config.from_dict(b["config"]), models=b["models"], filler=b["filler"],
                thresholds=b["thresholds"], meta=b["meta"])
