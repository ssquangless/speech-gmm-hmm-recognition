"""Các biểu đồ chuẩn dùng trong notebook (chỉ những biểu đồ có ý nghĩa với bài toán nhận dạng)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"up": "#1f77b4", "down": "#d62728", "left": "#ff7f0e", "right": "#2ca02c",
          "unknown": "#7f7f7f", "noise": "#9467bd"}


def setup_style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def plot_confusion(cm: np.ndarray, labels: Sequence[str], title: str = "Ma trận nhầm lẫn", ax=None):
    """Hiển thị đồng thời số lượng và tỷ lệ theo hàng (recall)."""
    cm = np.asarray(cm)
    rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rate, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{rate[i, j]:.1%}", ha="center", va="center",
                    color="white" if rate[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    return ax


def plot_learning_curves(history: List[dict], words: Sequence[str]):
    """Log-likelihood trung bình trên mỗi khung theo vòng EM: train (nét liền) và validation (nét đứt).

    Đường dọc: bắt đầu một lần khởi tạo lại bằng Viterbi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for w in words:
        h = [r for r in history if r["word"] == w]
        if not h:
            continue
        x = np.arange(1, len(h) + 1)
        for ax in axes:
            ax.plot(x, [r["train_ll"] for r in h], "-", color=COLORS.get(w), label=f"{w} train")
            ax.plot(x, [r["val_ll"] for r in h], "--", color=COLORS.get(w), label=f"{w} val")
    h0 = [r for r in history if r["word"] == words[0]]
    for i in range(1, len(h0)):
        if h0[i]["round"] != h0[i - 1]["round"]:
            axes[0].axvline(i + 0.5, color="k", ls=":", lw=1)
    axes[0].set_title("Toàn cảnh")
    axes[1].set_title("Phóng đại (bỏ 25% giá trị thấp nhất)")
    allv = np.array([r[k] for r in history for k in ("train_ll", "val_ll") if np.isfinite(r[k])])
    axes[1].set_ylim(np.percentile(allv, 25), allv.max() + 0.05 * (allv.max() - np.percentile(allv, 25)))
    for ax in axes:
        ax.set_xlabel("Vòng EM (cộng dồn qua các lần khởi tạo lại)")
        ax.set_ylabel("log-likelihood / khung")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_det(pos: np.ndarray, neg: np.ndarray, thr: Optional[float] = None, title: str = "Từ chối: FAR vs FRR"):
    from speech.evaluate import roc_points
    t, far, frr = roc_points(pos, neg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(far, 1 - frr, color="#1f77b4")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("FAR: tỷ lệ nhận nhầm từ khác/nhiễu")
    ax[0].set_ylabel("1 - FRR: tỷ lệ chấp nhận đúng lệnh")
    ax[0].set_title("Đường ROC")
    if thr is not None:
        ax[0].plot([np.mean(neg >= thr)], [np.mean(pos >= thr)], "ro", label=f"ngưỡng = {thr:.2f}")
        ax[0].legend()
    ax[1].hist(neg, bins=60, alpha=0.6, label="từ khác + nhiễu (cần từ chối)", color="#7f7f7f", density=True)
    ax[1].hist(pos, bins=60, alpha=0.6, label="4 lệnh (cần chấp nhận)", color="#1f77b4", density=True)
    if thr is not None:
        ax[1].axvline(thr, color="r", ls="--")
    ax[1].set_xlabel("LLR / khung = (LL từ thắng - LL mô hình nền) / T")
    ax[1].legend()
    ax[1].set_title("Phân bố điểm")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
