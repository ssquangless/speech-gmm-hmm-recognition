# Nhận dạng lệnh bằng HMM-GMM (up / down / left / right) — bản đơn giản, dùng hmmlearn

```
config.py                 mọi tham số (tiền xử lý, endpoint, mô hình, đường dẫn); được lưu kèm file mô hình
speech/
  endpoint.py             cắt đoạn chứa từ (năng lượng khung, tự thích nghi nền ồn)
  features.py             MFCC 39 chiều kiểu HTK + CMVN; utterance_features() là ĐIỂM VÀO DUY NHẤT
  dataset.py              nạp Speech Commands, chia theo người nói, cache đặc trưng, nhiễu nền
  hmm.py                  GMMHMM (hmmlearn) left-to-right: khởi tạo, huấn luyện lại bằng Viterbi, lưu/nạp
  recognizer.py           suy luận cho demo (từ, xác suất, từ chối)
  live.py                 thu mic + phát hiện câu nói (VAD) thời gian thực
  evaluate.py plots.py    chỉ số + biểu đồ chuẩn
notebooks/
  01_EDA.ipynb            khảo sát dữ liệu (độ lớn, thời lượng, người nói, endpoint, nền ồn/SNR so với mic của bạn)
  02_train.ipynb          chọn N/M trên validation, huấn luyện, đường cong hội tụ, mô hình nền, ngưỡng từ chối
  03_evaluate.ipynb       test theo người nói, đọc log-likelihood, từ chối, độ bền SNR/âm lượng, mic của bạn
tools/record_samples.py   thu mẫu bằng mic laptop
tools/check_hmmlearn.py   kiểm tra hmmlearn trên máy bạn có giữ đúng các thiết lập quan trọng không
main.py                   giao diện: hiển thị từ nhận được + waveform + năng lượng/ngưỡng VAD + xác suất 4 từ
tests/                    kiểm thử trên dữ liệu giả
```

## Chạy

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/check_hmmlearn.py          
python tests/test_pipeline.py           
```