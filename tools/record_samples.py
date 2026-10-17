"""Thu mẫu giọng của chính bạn bằng mic laptop để kiểm tra lệch miền dữ liệu.

Chạy:  python tools/record_samples.py --n 15
Kết quả: data/my_recordings/<từ>/<từ>_<k>.wav  (16 kHz, 16-bit, mono)
Sau đó chạy phần cuối của notebooks/03_evaluate.ipynb để xem độ chính xác trên chính mic của bạn.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import TARGET_WORDS  # noqa: E402

SR = 16000


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12)


def loudest_db(y, win=800):
    n = len(y) // win
    return max(db(y[i * win:(i + 1) * win]) for i in range(n)) if n else db(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=TARGET_WORDS)
    ap.add_argument("--n", type=int, default=15, help="số lần thu mỗi từ")
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--device", default=None, help="chỉ số/tên thiết bị (xem: python -m sounddevice)")
    ap.add_argument("--out", default=str(ROOT / "data" / "my_recordings"))
    a = ap.parse_args()
    if a.device is not None:
        sd.default.device = int(a.device) if a.device.isdigit() else a.device

    print("Đo nhiễu nền 1 giây: hãy giữ im lặng...")
    amb = sd.rec(SR, samplerate=SR, channels=1, dtype="float32"); sd.wait()
    noise_db = db(amb)
    print(f"Nhiễu nền: {noise_db:.1f} dBFS\n")

    for w in a.words:
        out = Path(a.out) / w
        out.mkdir(parents=True, exist_ok=True)
        print(f"=== Từ '{w}' ===")
        k = 0
        while k < a.n:
            input(f"[{k + 1}/{a.n}] Nhấn Enter rồi nói '{w}' ngay... ")
            y = sd.rec(int(a.seconds * SR), samplerate=SR, channels=1, dtype="float32"); sd.wait()
            y = y[:, 0]
            peak, snr = float(np.abs(y).max()), loudest_db(y) - noise_db
            msg = f"   đỉnh = {peak:.2f}, SNR ước lượng = {snr:.0f} dB"
            if peak >= 0.99:
                print(msg + "  -> BỊ MÉO (clipping), thu lại (giảm âm lượng mic)")
                continue
            if snr < 15:
                print(msg + "  -> quá nhỏ so với nhiễu, thu lại (nói to/gần mic hơn)")
                continue
            print(msg)
            wavfile.write(out / f"{w}_{int(time.time() * 1000)}_{k}.wav", SR, (y * 32767).astype(np.int16))
            k += 1
    print("\nXong. Các file nằm ở", a.out)


if __name__ == "__main__":
    main()
