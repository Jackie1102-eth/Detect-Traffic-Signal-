# Traffic Sign Recognition

Hệ thống nhận diện biển báo giao thông bằng Python + PyTorch + EfficientNet.

## Cấu trúc project

```
traffic_sign_project/
├── data/                  # Dữ liệu dataset (tải từ Roboflow)
├── src/
│   ├── dataset/           # Đọc & xử lý dữ liệu
│   ├── model/             # Kiến trúc và train model
│   └── utils/             # Hàm tiện ích dùng chung
├── scripts/               # Script chạy chính
├── outputs/
│   ├── checkpoints/       # File model đã train (.pth)
│   ├── logs/              # Log quá trình train
│   └── results/           # Ảnh kết quả predict
├── notebooks/             # Jupyter notebook thử nghiệm
├── config.py              # Toàn bộ cấu hình tập trung
└── requirements.txt
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

## Cách dùng

```bash
# 1. Tải dataset từ Roboflow
python scripts/download_data.py

# 2. Train model
python scripts/train.py

# 3. Nhận diện từ ảnh tĩnh
python scripts/predict_image.py --image path/to/image.jpg

# 4. Nhận diện real-time qua webcam
python scripts/predict_camera.py
```
