"""Nhận dạng lệnh bằng giọng nói (up / down / left / right) với HMM-GMM: hiển thị từ nhận được theo thời gian thực.

Chạy:  python main.py        Phím: SPACE = bật/tắt mic, R = đo lại nhiễu nền, ESC = thoát
Pipeline nhận dạng dùng chung với huấn luyện/đánh giá (xem thư mục speech/).
"""
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from speech.live import AudioStream, UtteranceDetector
from speech.recognizer import SpeechRecognizer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weight" / "hmm_gmm_speech_models.pkl"
SAVE_DEBUG_AUDIO = True                 # lưu mỗi câu nói vào debug_audio/ để nghe lại
DEBUG_DIR = ROOT / "debug_audio"
CHUNK_SEC = 0.05

if not MODEL_PATH.exists():
    raise SystemExit(f"Chưa có mô hình tại {MODEL_PATH}. Hãy chạy notebooks/02_train.ipynb trước.")

recognizer = SpeechRecognizer(str(MODEL_PATH))
SR = recognizer.sample_rate
stream = AudioStream(sr=SR, chunk_sec=CHUNK_SEC)
detector = UtteranceDetector(sr=SR, chunk_sec=CHUNK_SEC)
mic_ok, mic_msg = AudioStream.check_microphone()
if SAVE_DEBUG_AUDIO:
    DEBUG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------- giao diện
pygame.init()
W, H = 960, 600
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Nhan dang lenh bang giong noi (HMM-GMM)")
BG, PANEL, GRID = (245, 246, 250), (30, 39, 46), (70, 80, 90)
TEXT, GRAY, GREEN, RED, BLUE, YELLOW, ORANGE = (44, 62, 80), (150, 158, 165), (39, 174, 96), (231, 76, 60), (41, 128, 185), (241, 196, 15), (230, 126, 34)
f_big = pygame.font.SysFont("Arial", 110, bold=True)
f_mid = pygame.font.SysFont("Arial", 22, bold=True)
f_sml = pygame.font.SysFont("Arial", 14)
f_tiny = pygame.font.SysFont("Arial", 12)

rms_trace = deque([0.0] * 100, maxlen=100)          # 100 chunk x 50 ms = 5 giây
history = deque(maxlen=8)
shown_word, shown_ok, shown_info, shown_probs, shown_at = "", False, "Nhan SPACE de bat mic", {}, 0.0
listening = False


def start_mic():
    global listening, shown_info
    if not mic_ok:
        shown_info = "LOI: khong tim thay micro"
        return
    detector.reset()
    try:
        stream.start()
    except Exception as ex:                                  # noqa: BLE001 (thiết bị bận / không hỗ trợ 16 kHz...)
        shown_info = f"LOI mo micro: {ex}"[:110]
        return
    listening = True
    shown_info = "Dang do nhieu nen: giu im lang 1 giay..."


def stop_mic():
    global listening, shown_info
    stream.stop()
    detector.reset()
    listening = False
    shown_info = "Da tat mic (SPACE de bat)"


def handle(utt):
    global shown_word, shown_ok, shown_info, shown_probs, shown_at
    t0 = time.perf_counter()
    p = recognizer.predict(utt)
    dt = (time.perf_counter() - t0) * 1000
    if SAVE_DEBUG_AUDIO:
        from scipy.io import wavfile
        wavfile.write(str(DEBUG_DIR / f"{int(time.time() * 1000)}_{p.word or 'none'}.wav"), SR,
                      (np.clip(utt, -1, 1) * 32767).astype(np.int16))
    print(f"[{len(utt) / SR:.2f}s thu | {p.n_frames} khung | {dt:.0f} ms] {p.word} -> {p.reason} | LLR/khung {p.llr:.2f}"
          f" | " + ", ".join(f"{w} {v:.2f}" for w, v in p.probs.items()))
    shown_at, shown_probs = time.time(), p.probs
    if p.word is None:
        shown_word, shown_ok, shown_info = "", False, "Khong thay tieng noi"
    elif p.accepted:
        shown_word, shown_ok = p.word.upper(), True
        shown_info = f"p = {p.probs[p.word]:.2f}   LLR/khung = {p.llr:.2f}   ({dt:.0f} ms)"
        history.appendleft(f"{time.strftime('%H:%M:%S')}   {p.word.upper():<6} p={p.probs[p.word]:.2f}")
    else:
        shown_word, shown_ok = "?", False
        shown_info = f"Khong giong lenh nao (LLR {p.llr:.2f} < {recognizer.thr['min_llr_per_frame']:.2f}, doan: {p.word})"
        history.appendleft(f"{time.strftime('%H:%M:%S')}   -- bo qua ({p.word}, LLR {p.llr:.2f})")


def panel(rect, title):
    pygame.draw.rect(screen, PANEL, rect, border_radius=6)
    screen.blit(f_tiny.render(title, True, (200, 205, 210)), (rect.x + 8, rect.y + 5))


def draw_wave(rect):
    panel(rect, "Waveform (1 giay gan nhat)")
    v = stream.display[::100]
    mid, amp = rect.y + 22 + (rect.height - 22) // 2, (rect.height - 30) / 2
    peak = max(0.02, float(np.abs(v).max()) * 1.1)
    pts = [(rect.x + int(i / len(v) * rect.width), int(mid - x / peak * amp)) for i, x in enumerate(v)]
    pygame.draw.line(screen, GRID, (rect.x, mid), (rect.right, mid))
    if len(pts) > 1:
        pygame.draw.lines(screen, (46, 204, 113), False, pts, 2)
    screen.blit(f_tiny.render(f"thang do +/-{peak:.3f}", True, GRAY), (rect.right - 110, rect.y + 5))


def draw_energy(rect):
    panel(rect, "Nang luong (RMS) 5 giay gan nhat va nguong VAD")
    top = max(0.02, max(rms_trace) * 1.2, detector.start_thr * 1.5)
    y_of = lambda r: int(rect.bottom - 6 - min(r / top, 1.0) * (rect.height - 30))
    pts = [(rect.x + int(i / (len(rms_trace) - 1) * rect.width), y_of(r)) for i, r in enumerate(rms_trace)]
    pygame.draw.lines(screen, YELLOW if detector.state == detector.RECORDING else (120, 190, 255), False, pts, 2)
    for thr, col, lab in ((detector.start_thr, RED, "bat dau"), (detector.end_thr, ORANGE, "ket thuc")):
        y = y_of(thr)
        pygame.draw.line(screen, col, (rect.x, y), (rect.right, y), 1)
        screen.blit(f_tiny.render(lab, True, col), (rect.x + 6, y - 14))
    state = {"calibrating": "DO NHIEU NEN", "idle": "SAN SANG", "recording": "DANG THU"}[detector.state]
    # screen.blit(f_tiny.render(f"{state}   nen on {detector.noise_rms:.4f}", True, GRAY), (rect.right - 190, rect.y + 5))
    screen.blit(
        f_tiny.render(f"{state}   nen on {detector.noise_rms:.4f}", True, GRAY),
        (rect.x + 8, rect.bottom - 16)
    )

def draw_probs(rect):
    panel(rect, "Xac suat cua 4 tu (cau gan nhat)")
    words = recognizer.words
    for k, w in enumerate(words):
        y = rect.y + 30 + k * ((rect.height - 36) // len(words))
        p = shown_probs.get(w, 0.0)
        pygame.draw.rect(screen, (55, 66, 76), (rect.x + 70, y, rect.width - 130, 18), border_radius=4)
        if p > 0:
            pygame.draw.rect(screen, GREEN if (shown_ok and w.upper() == shown_word) else BLUE,
                             (rect.x + 70, y, int((rect.width - 130) * p), 18), border_radius=4)
        screen.blit(f_sml.render(w, True, (230, 235, 240)), (rect.x + 10, y))
        screen.blit(f_sml.render(f"{p:.2f}", True, (230, 235, 240)), (rect.right - 50, y))


clock, running = pygame.time.Clock(), True
if mic_ok:
    start_mic()
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
            stop_mic() if listening else start_mic()
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r and listening:
            detector.reset()
            shown_info = "Dang do nhieu nen: giu im lang 1 giay..."

    if listening:
        was_calib = detector.state == detector.CALIBRATING
        for chunk in stream.read_chunks():
            utt = detector.process(chunk)
            rms_trace.append(detector.rms)
            if was_calib and detector.state != detector.CALIBRATING:
                was_calib = False
                shown_info = f"San sang: hay noi up / down / left / right (nen on {detector.noise_rms:.4f})"
            if utt is not None:
                handle(utt)
        if detector.state == detector.RECORDING:
            shown_info = "Dang thu am..."

    screen.fill(BG)
    screen.blit(f_mid.render("Nhan dang lenh bang giong noi (HMM-GMM)", True, TEXT), (30, 18))
    screen.blit(f_sml.render("SPACE: bat/tat mic   R: do lai nhieu nen   ESC: thoat", True, GRAY), (30, 48))
    pygame.draw.circle(screen, GREEN if listening else GRAY, (W - 40, 32), 9)

    word_color = GREEN if shown_ok else GRAY
    txt = f_big.render(shown_word or "...", True, word_color if shown_word else (200, 204, 208))
    screen.blit(txt, txt.get_rect(center=(300, 190)))
    screen.blit(f_sml.render(shown_info, True, TEXT), (30, 270))

    screen.blit(f_mid.render("Lich su", True, TEXT), (30, 310))
    for k, line in enumerate(history):
        screen.blit(f_sml.render(line, True, TEXT if k else GREEN), (30, 345 + 28 * k))

    draw_wave(pygame.Rect(580, 80, 350, 150))
    draw_energy(pygame.Rect(580, 245, 350, 150))
    draw_probs(pygame.Rect(580, 410, 350, 170))
    pygame.display.flip()
    clock.tick(30)

stream.stop()
pygame.quit()
