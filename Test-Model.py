from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from PIL import Image
import os

# 1. Đọc file Test.csv
# Đảm bảo file Test.csv nằm cùng thư mục với file code này
y_test_data = pd.read_csv('Test.csv')

# Lấy nhãn thực tế (ClassId) và đường dẫn ảnh (Path)
labels = y_test_data["ClassId"].values
imgs = y_test_data["Path"].values

data = []

print("Đang kiểm tra độ chính xác trên tập dữ liệu Test...")

# 2. Lặp qua danh sách ảnh trong tập Test
for img in imgs:
    try:
        # Quan trọng: Mở ảnh theo đường dẫn từ CSV
        # Nếu thư mục Test nằm trong thư mục gốc, OpenCV/PIL cần đường dẫn đúng
        image = Image.open(img)
        image = image.resize((30, 30)) # Phải khớp kích thước lúc train (30x30)
        data.append(np.array(image))
    except Exception as e:
        print(f"Không thể mở ảnh {img}: {e}")

# Chuyển dữ liệu test về dạng mảng numpy
X_test_batch = np.array(data)

# 3. Dự đoán bằng mô hình đã huấn luyện
# Giả sử mô hình của bạn đang được lưu trong biến 'model'
# Nếu chạy file độc lập, hãy dùng: model = load_model('traffic_classifier.h5')
pred_prob = model.predict(X_test_batch)
pred = np.argmax(pred_prob, axis=-1)

# 4. Tính toán và in ra độ chính xác
score = accuracy_score(labels, pred)
print("-" * 30)
print(f"ĐỘ CHÍNH XÁC TRÊN TẬP TEST: {score * 100:.2f}%")
print("-" * 30)