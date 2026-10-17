"""Kiểm thử đầu-cuối trên dữ liệu giả: đặc trưng, endpoint, chia theo người nói, huấn luyện (hmmlearn), nhận dạng, VAD.
Chạy:  python tests/test_pipeline.py   (cần hmmlearn đã cài)"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, TARGET_WORDS
from speech.dataset import build_feature_sets, get_items, read_wav, verify_speaker_disjoint
from speech.endpoint import find_speech_segment
from speech.evaluate import score_matrix
from speech.features import compute_features
from speech.hmm import filler_loglik, save_bundle, train_all, train_filler
from speech.live import UtteranceDetector
from speech.recognizer import SpeechRecognizer
from synth_dataset import make_dataset


def test_features_and_endpoint():
    cfg = Config()
    rng = np.random.default_rng(0)
    y = rng.standard_normal(16000) * 0.1
    f = compute_features(y, 16000, cfg.features)
    assert f.shape[1] == 39 and f.shape[0] == 1 + (16000 - 400) // 160
    assert np.abs(f - compute_features(y * 0.1, 16000, cfg.features)).max() < 1e-3     # bất biến theo âm lượng

    sr = 16000
    y = rng.standard_normal(sr) * 0.003
    t = np.arange(int(0.3 * sr)) / sr
    y[6000:6000 + len(t)] += 0.2 * np.sin(2 * np.pi * 500 * t) * np.hanning(len(t))
    seg = find_speech_segment(y, sr, cfg.endpoint)
    assert seg is not None and 4000 < seg[0] < 6200 and 10200 < seg[1] < 12500, seg
    assert find_speech_segment(rng.standard_normal(sr) * 0.003, sr, cfg.endpoint) is None
    assert find_speech_segment(np.zeros(sr), sr, cfg.endpoint) is None


def test_utterance_detector():
    sr, ch = 16000, 800
    rng = np.random.default_rng(5)
    det = UtteranceDetector(sr=sr, chunk_sec=0.05)
    sig = rng.standard_normal(sr * 5) * 0.002
    for a in (2.0, 3.6):
        s = int(a * sr)
        t = np.arange(int(0.35 * sr)) / sr
        sig[s:s + len(t)] += 0.05 * np.sin(2 * np.pi * 600 * t) * np.hanning(len(t))
    found = [len(u) / sr for u in (det.process(sig[i:i + ch].astype(np.float32))
                                   for i in range(0, len(sig) - ch, ch)) if u is not None]
    assert len(found) == 2 and all(0.6 < d < 1.5 for d in found), found


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_dataset(Path(tmp) / "sc", n_speakers=70, utt_per_word=3)
        cfg = Config()
        cfg.data.data_root, cfg.data.cache_dir = str(root), str(Path(tmp) / "cache")
        cfg.data.filler_per_word, cfg.data.noise_segments = 40, 60
        cfg.model.n_states = {w: 4 for w in TARGET_WORDS}
        cfg.model.n_mix, cfg.model.n_iter, cfg.model.realign_rounds, cfg.model.filler_mix = 2, 4, 1, 4

        items = get_items(cfg, TARGET_WORDS)
        assert not any(verify_speaker_disjoint(items).values())          # chia theo người nói, không trùng
        data = build_feature_sets(cfg, n_jobs=1, verbose=False)
        tr = {w: data["sets"]["train"][w] for w in TARGET_WORDS}
        va = {w: data["sets"]["val"][w] for w in TARGET_WORDS}
        models, hist = train_all(tr, va, cfg, n_jobs=1)
        for m in models.values():                                        # left-to-right được giữ sau huấn luyện
            assert np.all(np.tril(m.transmat_, -1) == 0) and np.allclose(m.startprob_[0], 1.0)
        assert all(np.isfinite(h["val_ll"]) for h in hist)

        filler = train_filler(list(data["unknown"]["train"]) + list(data["noise"]["train"]), cfg.model.filler_mix)
        seqs, y = [], []
        for i, w in enumerate(TARGET_WORDS):
            seqs += [np.asarray(x, np.float64) for x in data["sets"]["test"][w]]
            y += [i] * len(data["sets"]["test"][w])
        acc = np.mean(score_matrix(models, seqs).argmax(1) == np.array(y))
        print("accuracy trên dữ liệu giả:", round(float(acc), 3), "n_test =", len(y))
        assert acc > 0.9

        path = Path(tmp) / "w" / "m.pkl"
        save_bundle(path, cfg, models, filler, thresholds=dict(min_llr_per_frame=-1e9, temperature=5.0))
        rec = SpeechRecognizer(str(path))
        item = [i for i in items if i["split"] == "test" and i["word"] == "left"][0]
        p = rec.predict(read_wav(item["path"]))
        assert p.word == "left" and p.accepted and abs(sum(p.probs.values()) - 1) < 1e-6, p
        assert rec.predict(np.random.default_rng(3).standard_normal(16000) * 0.003).reason == "no_speech"


if __name__ == "__main__":
    test_features_and_endpoint(); print("features/endpoint ok")
    test_utterance_detector(); print("detector ok")
    test_full_pipeline(); print("pipeline ok")
