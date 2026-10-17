"""Kiểm tra nhanh (vài giây) rằng hmmlearn trên MÁY BẠN giữ được các thiết lập quan trọng mà dự án dựa vào.

Chạy:  python tools/check_hmmlearn.py
Nếu có dòng LỖI, đừng huấn luyện: gửi toàn bộ đầu ra này cho người hỗ trợ.
"""
import sys
from pathlib import Path

import joblib
import numpy as np

import hmmlearn
from hmmlearn.hmm import GMMHMM

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(("OK   " if cond else "LỖI "), name, detail)


def sample_lr(rng, means, a_stay=0.85):
    out = []
    for s in range(len(means)):
        for _ in range(int(rng.geometric(1 - a_stay))):
            out.append(means[s] + 0.7 * rng.standard_normal(means.shape[1]))
    return np.array(out)


def main():
    print("hmmlearn", getattr(hmmlearn, "__version__", "?"))
    rng = np.random.default_rng(0)
    N, M, D = 4, 2, 5
    true = rng.standard_normal((N, D)) * 3
    seqs = [sample_lr(rng, true) for _ in range(150)]
    X, lengths = np.concatenate(seqs), [len(s) for s in seqs]

    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i], A[i, i + 1] = 0.6, 0.4
    A[N - 1, N - 1] = 1.0
    start = np.eye(N)[0]
    mu = true[:, None, :] + 0.1 * rng.standard_normal((N, M, D))
    m = GMMHMM(n_components=N, n_mix=M, covariance_type="diag", min_covar=1e-2, n_iter=1, tol=0.0,
               init_params="", params="stmcw", random_state=0)
    m.startprob_, m.transmat_ = start, A
    m.weights_, m.means_, m.covars_ = np.full((N, M), 1 / M), mu, np.ones((N, M, D))

    ll = []
    try:
        for _ in range(6):
            m.fit(X, lengths)
            ll.append(float(m.monitor_.history[-1]))
        check("1. đặt tham số bằng tay rồi fit(n_iter=1) nhiều lần không lỗi", True)
    except Exception as e:                                           # noqa: BLE001
        check("1. đặt tham số bằng tay rồi fit(n_iter=1) nhiều lần không lỗi", False, repr(e))
        sys.exit(1)

    check("2. transmat_ giữ nguyên số 0 (left-to-right)",
          np.all(m.transmat_[np.tril_indices(N, -1)] == 0) and np.all(np.triu(m.transmat_, 2) == 0))
    check("3. startprob_ giữ nguyên [1,0,...,0]", np.allclose(m.startprob_, np.eye(N)[0]))
    err = np.abs(m.means_.mean(axis=1) - true).max()
    check("4. khởi tạo của ta không bị ghi đè (means_ vẫn gần giá trị đặt)", err < 1.5, f"(lệch tối đa {err:.2f})")
    check("5. fit(n_iter=1) lặp lại = tiếp tục huấn luyện (LL không giảm)",
          all(b >= a - 1e-6 * abs(a) for a, b in zip(ll, ll[1:])), f"(LL: {ll[0]:.0f} -> {ll[-1]:.0f})")
    check("6. kích thước tham số đúng (weights_/means_/covars_)",
          m.weights_.shape == (N, M) and m.means_.shape == (N, M, D) and m.covars_.shape == (N, M, D))
    check("7. sàn phương sai min_covar được áp dụng", m.covars_.min() >= 0.99e-2, f"(nhỏ nhất {m.covars_.min():.4f})")
    s0 = seqs[0]
    sc, st = m.score(s0), m.predict(s0)
    check("8. score() và predict() (Viterbi) chạy trên một chuỗi", np.isfinite(sc) and len(st) == len(s0))
    tmp = Path("_check_tmp.pkl")
    joblib.dump(m, tmp)
    m2 = joblib.load(tmp)
    tmp.unlink()
    check("9. lưu/nạp bằng joblib giữ nguyên kết quả", abs(m2.score(s0) - sc) < 1e-6)

    print("\nTẤT CẢ ĐẠT: có thể huấn luyện." if all(OK) else "\nCÓ LỖI: hãy gửi đầu ra này để được hỗ trợ trước khi huấn luyện.")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
