import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import datetime

# --- 1. HÀM VẼ BIỂU ĐỒ HIỆU NĂNG ---
def plot_performance(history):
    plt.figure(figsize=(12, 5))

    # Đồ thị Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Đồ thị Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# --- 2. NẠP DỮ LIỆU HUẤN LUYỆN (TRAIN) ---
data = []
labels = []
classes = 43
cur_path = os.getcwd()

print("Bắt đầu nạp dữ liệu Train... Vui lòng đợi.")
for i in range(classes):
    # Đường dẫn chuẩn cho máy tính cá nhân
    path = os.path.join(cur_path, 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Lỗi nạp ảnh {a}: {e}")

# Chuyển sang Numpy array
data = np.array(data)
labels = np.array(labels)
print(f"Tổng số ảnh nạp được: {data.shape[0]}")

# --- 3. CHIA DỮ LIỆU VÀ TIỀN XỬ LÝ ---
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Chuyển nhãn sang One-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# --- 4. XÂY DỰNG MÔ HÌNH CNN CẢI TIẾN ---
model = Sequential()
# Lớp 1 & 2
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.15))

# Lớp 3 & 4
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.20))

# Lớp kết nối đầy đủ (Fully Connected)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(43, activation='softmax'))

# Biên dịch
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 5. HUẤN LUYỆN (TRAINING) ---
# Nếu máy bạn có GPU NVIDIA đã cài CUDA, nó sẽ tự dùng. 
# Nếu không nó sẽ dùng CPU, bạn không cần lo dòng 'tf.device'.
epochs = 35 # Bạn có thể tăng lên 35 nếu máy mạnh
batch_size = 64

print(f"Bắt đầu huấn luyện {epochs} Epochs...")
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Lưu mô hình ngay sau khi train xong
model.save('traffic_classifier.h5')
print("--- Đã lưu mô hình: traffic_classifier.h5 ---")

# Vẽ biểu đồ hiệu năng
plot_performance(history)

# --- 6. ĐÁNH GIÁ TRÊN TẬP TEST THỰC TẾ (SỬ DỤNG TEST.CSV) ---
print("Đang đánh giá trên tập dữ liệu Test thực tế...")
try:
    test_csv = pd.read_csv('Test.csv')
    labels_test = test_csv["ClassId"].values
    imgs_test = test_csv["Path"].values
    
    data_test = []
    for img_path in imgs_test:
        # Lưu ý: img_path trong CSV thường là 'Test/00001.png'
        image = Image.open(img_path)
        image = image.resize((30, 30))
        data_test.append(np.array(image))
    
    X_test_final = np.array(data_test)
    
    # Dự đoán
    pred = np.argmax(model.predict(X_test_final), axis=-1)

    # Tính độ chính xác cuối cùng
    from sklearn.metrics import accuracy_score
    final_acc = accuracy_score(labels_test, pred)
    print(f"ĐỘ CHÍNH XÁC CUỐI CÙNG TRÊN TẬP TEST: {final_acc * 100:.2f}%")
except Exception as e:
    print(f"Lỗi khi đánh giá tập Test: {e}")
    print("Mẹo: Hãy đảm bảo file Test.csv và thư mục Test nằm đúng chỗ.")