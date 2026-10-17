"""Thu âm thời gian thực + phát hiện câu nói (VAD) theo năng lượng, tự thích nghi với nhiễu nền.

``UtteranceDetector`` không phụ thuộc phần cứng (chỉ nhận từng chunk numpy) nên kiểm thử được.
Endpoint chính xác được tinh chỉnh lại sau đó bởi ``speech.endpoint.find_speech_segment``
(cùng hàm với lúc huấn luyện) bên trong ``SpeechRecognizer.predict``.
"""
from __future__ import annotations

import queue
from collections import deque
from typing import List, Optional

import numpy as np


class UtteranceDetector:
    CALIBRATING, IDLE, RECORDING = "calibrating", "idle", "recording"

    def __init__(self, sr: int = 16000, chunk_sec: float = 0.05, preroll_sec: float = 0.25,
                 hangover_sec: float = 0.4, max_sec: float = 2.0, min_voiced_sec: float = 0.1,
                 start_ratio: float = 3.0, end_ratio: float = 1.8, min_abs_rms: float = 0.002,
                 calib_sec: float = 1.0):
        self.sr = sr
        self.chunk_sec = chunk_sec
        self.preroll = deque(maxlen=max(1, int(round(preroll_sec / chunk_sec))))
        self.hangover_n = max(1, int(round(hangover_sec / chunk_sec)))
        self.max_n = int(round(max_sec / chunk_sec))
        self.min_voiced_n = max(1, int(round(min_voiced_sec / chunk_sec)))
        self.start_ratio, self.end_ratio, self.min_abs = start_ratio, end_ratio, min_abs_rms
        self.calib_n = max(1, int(round(calib_sec / chunk_sec)))
        self.reset()

    def reset(self) -> None:
        self.state = self.CALIBRATING
        self._calib: List[float] = []
        self.noise_rms = 0.0
        self.rms = 0.0
        self.segment: List[np.ndarray] = []
        self.voiced = 0
        self.silent = 0
        self.preroll.clear()

    @property
    def start_thr(self) -> float:
        return max(self.min_abs, self.noise_rms * self.start_ratio)

    @property
    def end_thr(self) -> float:
        return max(self.min_abs * 0.6, self.noise_rms * self.end_ratio)

    def process(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Nạp một chunk; trả về mảng âm thanh của câu nói khi câu kết thúc, ngược lại None."""
        rms = float(np.sqrt(np.mean(np.asarray(chunk, dtype=np.float64) ** 2)))
        self.rms = rms
        if self.state == self.CALIBRATING:
            self._calib.append(rms)
            if len(self._calib) >= self.calib_n:
                self.noise_rms = float(np.median(self._calib))
                self.state = self.IDLE
            return None

        if self.state == self.IDLE:
            self.preroll.append(chunk)
            if rms > self.start_thr:
                self.state = self.RECORDING
                self.segment = list(self.preroll)
                self.voiced, self.silent = 1, 0
            else:                                           # cập nhật nhiễu nền chậm
                self.noise_rms = 0.98 * self.noise_rms + 0.02 * rms
            return None

        # RECORDING
        self.segment.append(chunk)
        if rms > self.start_thr:
            self.voiced += 1
        self.silent = 0 if rms > self.end_thr else self.silent + 1
        if self.silent >= self.hangover_n or len(self.segment) >= self.max_n:
            utt = np.concatenate(self.segment)
            ok = self.voiced >= self.min_voiced_n
            self.segment, self.voiced, self.silent = [], 0, 0
            self.preroll.clear()
            self.state = self.IDLE
            return utt if ok else None
        return None


class AudioStream:
    """Luồng thu mic (sounddevice) -> hàng đợi các chunk float32."""

    def __init__(self, sr: int = 16000, chunk_sec: float = 0.05, display_sec: float = 1.0):
        self.sr = sr
        self.chunk = int(sr * chunk_sec)
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self.display = np.zeros(int(sr * display_sec), dtype=np.float32)

    @staticmethod
    def check_microphone():
        try:
            import sounddevice as sd
            devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
            if not devs:
                return False, "Khong tim thay thiet bi Micro (Input Device)!"
            return True, "Micro da san sang."
        except Exception as e:                              # pragma: no cover
            return False, f"Loi kiem tra Micro: {e}"

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.q.put(indata[:, 0].copy())

    def start(self) -> None:
        import sounddevice as sd
        if self.stream is not None:
            return
        self.drain()
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     blocksize=self.chunk, callback=self._callback)
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.drain()

    def drain(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()

    def read_chunks(self) -> List[np.ndarray]:
        out = []
        while not self.q.empty():
            c = self.q.get_nowait()
            out.append(c)
            n = len(c)
            self.display = np.roll(self.display, -n)
            self.display[-n:] = c
        return out
