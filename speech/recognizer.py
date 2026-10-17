"""Bộ nhận dạng cho demo: waveform -> Prediction (từ, xác suất, lý do từ chối)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from speech.features import utterance_features
from speech.hmm import filler_loglik, load_bundle


@dataclass
class Prediction:
    word: Optional[str]                    # từ thắng (None nếu không có tiếng nói)
    accepted: bool
    reason: str                            # ok | no_speech | low_llr
    probs: Dict[str, float] = field(default_factory=dict)    # xác suất hiệu chuẩn của 4 từ
    scores: Dict[str, float] = field(default_factory=dict)   # tổng log-likelihood
    llr: float = float("nan")              # (LL từ thắng - LL nền) / số khung
    n_frames: int = 0
    segment: Optional[Tuple[int, int]] = None


class SpeechRecognizer:
    def __init__(self, model_path: str = "weight/hmm_gmm_speech_models.pkl"):
        b = load_bundle(model_path)
        self.cfg, self.models, self.filler, self.thr = b["config"], b["models"], b["filler"], b["thresholds"]
        self.words = list(self.models)

    @property
    def sample_rate(self) -> int:
        return self.cfg.audio.sample_rate

    def predict(self, y: np.ndarray) -> Prediction:
        feat, seg = utterance_features(np.asarray(y, np.float64), self.cfg, return_segment=True)
        if feat is None:
            return Prediction(None, False, "no_speech")
        s = np.array([self.models[w].score(feat) for w in self.words])
        z = (s - s.max()) / self.thr.get("temperature", 1.0)
        p = np.exp(z) / np.exp(z).sum()
        k = int(np.argmax(s))
        llr = float((s[k] - filler_loglik(self.filler, [feat])[0]) / len(feat))
        ok = llr >= self.thr.get("min_llr_per_frame", -np.inf)
        return Prediction(self.words[k], ok, "ok" if ok else "low_llr", dict(zip(self.words, p)),
                          dict(zip(self.words, s)), llr, len(feat), seg)
