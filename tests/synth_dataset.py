"""Sinh bộ dữ liệu GIẢ có cấu trúc giống Speech Commands (để kiểm thử toàn bộ pipeline khi không có dữ liệu thật)."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 16000
# mỗi "từ" là chuỗi âm vị: (formant1, formant2, loại) ; loại: v = hữu thanh, f = ồn (xát)
WORDS = {
    "up":    [(700, 1200, "v"), (3500, 3500, "f")],
    "down":  [(350, 1800, "v"), (700, 1100, "v"), (300, 1500, "v")],
    "left":  [(400, 1000, "v"), (600, 1900, "v"), (3200, 3200, "f"), (2000, 2000, "f")],
    "right": [(500, 1400, "v"), (800, 1300, "v"), (350, 2200, "v")],
    "yes":   [(300, 2300, "v"), (550, 1800, "v"), (3800, 3800, "f")],
    "no":    [(450, 1100, "v"), (300, 800, "v")],
    "stop":  [(3600, 3600, "f"), (650, 1300, "v"), (1800, 1800, "f")],
    "go":    [(450, 900, "v"), (300, 700, "v")],
}


def synth_word(word, f0, scale, rng):
    t_all = []
    for (f1, f2, kind) in WORDS[word]:
        d = rng.uniform(0.07, 0.13)
        n = int(d * SR)
        t = np.arange(n) / SR
        env = np.hanning(n)
        if kind == "v":
            buzz = 0.5 * (1 + np.cos(2 * np.pi * f0 * t))
            x = (np.sin(2 * np.pi * f1 * scale * t) + 0.6 * np.sin(2 * np.pi * f2 * scale * t)) * buzz
        else:
            x = rng.standard_normal(n)
            # lọc băng thông thô quanh f1 bằng điều chế
            x = x * np.cos(2 * np.pi * f1 * scale * t) * 0.8
        t_all.append(x * env)
    return np.concatenate(t_all)


def make_dataset(root, n_speakers=60, utt_per_word=3, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "_background_noise_").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["white_noise", "pink_noise", "running_tap"]):
        n = rng.standard_normal(SR * 20)
        if name != "white_noise":
            n = np.cumsum(n) * 0.02 if name == "pink_noise" else np.convolve(n, np.ones(8) / 8, "same")
            n = n - np.convolve(n, np.ones(4000) / 4000, "same")
        n = n / (np.abs(n).max() + 1e-9) * 0.2
        wavfile.write(root / "_background_noise_" / f"{name}.wav", SR, (n * 32767).astype(np.int16))
    val, test = [], []
    for s in range(n_speakers):
        spk = hashlib.sha1(f"spk{s}".encode()).hexdigest()[:8]
        f0 = rng.uniform(90, 240)
        scale = rng.uniform(0.9, 1.15)
        h = int(spk, 16) % 100
        split = "test" if h < 15 else "val" if h < 30 else "train"
        for w in WORDS:
            (root / w).mkdir(exist_ok=True)
            for k in range(utt_per_word):
                x = synth_word(w, f0 * rng.uniform(0.95, 1.05), scale, rng)
                x = x / (np.abs(x).max() + 1e-9) * rng.uniform(0.15, 0.7)
                clip = np.zeros(SR)
                start = int(rng.uniform(0.15, 0.4) * SR)
                end = min(SR, start + len(x))
                clip[start:end] = x[: end - start]
                noise = rng.standard_normal(SR) * rng.uniform(0.002, 0.012)
                clip = clip + noise
                fn = f"{spk}_nohash_{k}.wav"
                wavfile.write(root / w / fn, SR, (np.clip(clip, -1, 1) * 32767).astype(np.int16))
                if split == "test":
                    test.append(f"{w}/{fn}")
                elif split == "val":
                    val.append(f"{w}/{fn}")
    (root / "validation_list.txt").write_text("\n".join(val))
    (root / "testing_list.txt").write_text("\n".join(test))
    return root
