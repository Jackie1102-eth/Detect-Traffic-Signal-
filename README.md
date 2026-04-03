# 🚦 Traffic Sign Detector — Hướng dẫn cài đặt & chạy

## Giới thiệu
Ứng dụng nhận diện biển báo giao thông bằng mô hình AI (TensorFlow/Keras).  
Hỗ trợ 2 chế độ:
-GUI — mở ảnh từ máy, nhận diện với top 3 dự đoán
-Camera — nhận diện realtime qua webcam

---

## Cấu trúc thư mục

```
traffic-sign-detector/
├── Meta/                   ← Metadata dataset
├── Test/                   ← Ảnh test
├── Train/                  ← Ảnh huấn luyện
├── Camera.py               ← Nhận diện realtime qua webcam
├── GUI.py                  ← Giao diện nhận diện từ ảnh
├── Train.py                ← Script huấn luyện model
├── labels.json             ← Tên các class biển báo
├── traffic_classifier.h5   ← Model đã huấn luyện
├── Meta.csv                ← Thông tin metadata
├── Test.csv                ← Nhãn tập test
├── Train.csv               ← Nhãn tập train
├── requirements.txt        ← Danh sách thư viện
└── README.md
```

---
## Cài đặt

### Bước 1 — Clone hoặc tải source code

```bash
git clone https://github.com/Jackie1102-eth/Detect-Traffic-Signal-.git
```

### Bước 2 — Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
```

### Bước 3 — Cài thư viện

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

### Chế độ GUI — nhận diện từ ảnh

```bash
python GUI.py
```

| Bước | Thao tác |
|---|---|
| 1 | Nhấn 📂 Mở ảnh → chọn ảnh biển báo |
| 2 | Nhấn 🔍 Nhận diện → chờ kết quả |
| 3 | Đọc kết quả: tên biển báo + độ tự tin top 3 |
| 4 | Nhấn 🗑 Xoá để thử ảnh khác |

Định dạng ảnh hỗ trợ: `.png` `.jpg` `.jpeg` `.bmp` `.ppm`

---

### Chế độ Camera — nhận diện realtime qua webcam

```bash
python Camera.py
```

- Webcam sẽ tự bật, kết quả nhận diện hiển thị trực tiếp trên khung hình
- Nhấn Q để thoát

> ⚠️ Cần có webcam hoặc camera USB. Nếu máy có nhiều camera, chỉnh `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)` trong `Camera.py`.

---

## Huấn luyện lại model

> Chỉ cần thực hiện nếu muốn train lại từ đầu. File `traffic_classifier.h5` đã có sẵn.

```bash
python Train.py
```

- Dữ liệu train lấy từ thư mục `Train/` và file `Train.csv`
- Sau khi train xong, file `traffic_classifier.h5` sẽ được ghi đè

---

## Ý nghĩa màu sắc độ tự tin (GUI)

| Màu | Ngưỡng | Ý nghĩa |
|---|---|---|
| 🟢 Xanh lá | ≥ 70% | Tự tin cao |
| 🟡 Vàng | 40–69% | Tự tin trung bình |
| 🔴 Đỏ | 30–39% | Tự tin thấp |
| ❓ Unknown | < 30% | Không nhận ra — ảnh không phải biển báo |

Để chỉnh ngưỡng Unknown, mở `GUI.py` và sửa:

```python
UNKNOWN_THRESHOLD = 0.30  # Tăng nếu hay nhận sai, giảm nếu Unknown quá nhiều
```
## Dataset
Project sử dụng **GTSRB — German Traffic Sign Recognition Benchmark** với 42 class biển báo.  
Thông tin chi tiết xem tại `Meta.csv` và thư mục `Meta/`.