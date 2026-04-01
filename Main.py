import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# 1. KHỞI TẠO CÁC BIẾN
data = []      
labels = []    
classes = 43   
cur_path = os.getcwd() 

# 2. ĐỌC DỮ LIỆU 
print("Đang nạp ảnh từ thư mục Train... Vui lòng đợi.")
for i in range(classes):
    path = os.path.join(cur_path, 'Train', str(i))
    images = os.listdir(path)
    
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30,30)) 
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Lỗi nạp ảnh: {a}")

# Chuyển sang mảng Numpy
data = np.array(data)
labels = np.array(labels)

# 3. CHIA DỮ LIỆU
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# 4. XÂY DỰNG MÔ HÌNH CNN
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# 5. BIÊN DỊCH VÀ HUẤN LUYỆN
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Bắt đầu huấn luyện (Epochs: 15)...")
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# 6. LƯU MÔ HÌNH
model.save("traffic_classifier.h5")
print("--- THÀNH CÔNG! Đã tạo file traffic_classifier.h5 ---")

# 7. VẼ ĐỒ THỊ ĐỂ KIỂM TRA
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()