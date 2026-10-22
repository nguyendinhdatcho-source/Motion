# VSL – nhánh Motion (DAT301m)

Nhận dạng ngôn ngữ ký hiệu Việt Nam (100 lớp) chỉ từ chuyển động giữa các frame.
Mô hình chính **M0**: frame difference → MobileNetV2 (ImageNet) cho từng bước → trung bình theo thời gian → Dense 100.

| Cấu hình | Val Macro-F1 | Test Macro-F1 (CI 95%) | Test Top-1 |
|---|---|---|---|
| m0_diff_avg | 0.821 | 0.862 (0.841–0.918) | 94.1% |
| m1_flow_avg | 0.752 | 0.782 (0.758–0.846) | 87.9% |
| m2_diff_tconv | 0.791 | 0.865 (0.836–0.916) | 92.8% |

## Cấu trúc

- `make_splits.py` – tạo `splits.csv` từ `data/{train,valid,test}/<lớp>/*.mp4` theo `label_mapping.pkl`
- `precompute_motion.py` – tính trước frame difference và optical flow (.npz)
- `train_motion.py` – train và đánh giá (`--mode diff|flow`, `--temporal avg|tconv|gru`)
- `vsl_motion_local.ipynb` – chạy toàn bộ pipeline và bảng so sánh
- `demo_m0.py` – demo Streamlit cho M0
- `vsl_motion_work/` – `splits.csv` và kết quả từng run (`metrics.json`, báo cáo, xác suất dự đoán)

Dữ liệu (`data/`) và file chuyển động tiền tính không có trong repo. Chỉ có trọng số của M0.

## Chạy

```bash
pip install tensorflow opencv-python pandas scikit-learn matplotlib streamlit
# đặt dữ liệu vào data/ (train/, valid/, test/, label_mapping.pkl), rồi chạy notebook vsl_motion_local.ipynb
streamlit run demo_m0.py   # demo M0, cần data/test để chọn video có sẵn
```
