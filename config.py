"""Cấu hình trung tâm: mọi tham số tiền xử lý / mô hình nằm ở đây và được lưu kèm file mô hình (.pkl),
nên demo luôn dùng đúng tham số đã dùng khi huấn luyện."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_WORDS = ["up", "down", "left", "right"]


@dataclass
class AudioConfig:
    sample_rate: int = 16000          # Speech Commands: 16 kHz, 16-bit, mono


@dataclass
class FeatureConfig:
    """MFCC kiểu HTK/Kaldi cho tiếng nói 16 kHz."""
    preemphasis: float = 0.97
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 26
    fmin: float = 20.0
    fmax: float = 8000.0
    n_ceps: int = 13                  # c0..c12
    lifter: int = 22
    delta_window: int = 2
    use_delta: bool = True
    use_delta2: bool = True
    norm: str = "cmvn"                # "cmvn" | "cmn" | "none" (theo từng utterance)
    log_floor: float = 1e-10


@dataclass
class EndpointConfig:
    """Cắt đoạn chứa từ (dùng giống hệt khi train / đánh giá / chạy thật)."""
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_percentile: float = 10.0
    peak_percentile: float = 98.0
    min_dynamic_db: float = 10.0      # peak - noise nhỏ hơn => không có tiếng nói
    snr_floor_db: float = 8.0
    peak_range_db: float = 35.0
    max_gap_ms: float = 200.0
    min_run_ms: float = 40.0
    pad_ms: float = 80.0
    min_duration_ms: float = 150.0


@dataclass
class ModelConfig:
    n_states: dict = field(default_factory=lambda: {"up": 5, "down": 6, "left": 6, "right": 6})
    n_mix: int = 4                    # số Gaussian mỗi trạng thái
    min_covar: float = 1e-2           # sàn phương sai (đặc trưng đã CMVN nên phương sai toàn cục ~1)
    n_iter: int = 12                  # số vòng EM tối đa mỗi lần khởi tạo (early stopping theo validation)
    patience: int = 2                 # dừng nếu val LL không tăng sau ngần ấy vòng
    realign_rounds: int = 2           # số lần "khởi tạo lại bằng Viterbi" (chỉ nhận nếu val LL tốt hơn)
    filler_mix: int = 16              # số Gaussian của mô hình nền (từ chối từ khác/nhiễu)
    seed: int = 0


@dataclass
class DataConfig:
    data_root: str = os.environ.get("SC_DATA_ROOT", str(ROOT / "data" / "speech_commands_v0.02"))
    cache_dir: str = str(ROOT / "cache")
    weights_dir: str = str(ROOT / "weight")
    url: str = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    filler_per_word: int = 150        # số clip train / từ "unknown" để huấn luyện mô hình nền
    noise_segments: int = 600


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(audio=AudioConfig(**d["audio"]), features=FeatureConfig(**d["features"]),
                   endpoint=EndpointConfig(**d["endpoint"]), model=ModelConfig(**d["model"]),
                   data=DataConfig(**d["data"]))

    def frontend_hash(self) -> str:
        """Hash phần tiền xử lý (đặt tên cache đặc trưng)."""
        blob = json.dumps({"a": asdict(self.audio), "f": asdict(self.features),
                           "e": asdict(self.endpoint)}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:10]
