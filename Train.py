import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import json

# --- 1. LOAD DATA ---
data = []
labels = []
classes = 43
cur_path = os.getcwd()

print("Đang load dữ liệu...")

for i in range(classes):
    path = os.path.join(cur_path, 'Train', str(i))
    images = os.listdir(path)

    for img_name in images:
        try:
            img = Image.open(os.path.join(path, img_name)).convert("RGB")
            img = img.resize((30, 30))
            img = np.array(img)
            data.append(img)
            labels.append(i)
        except:
            pass

# Convert + Normalize
data = np.array(data) / 255.0
labels = np.array(labels)

# Shuffle
data, labels = shuffle(data, labels)

print("Tổng số ảnh:", data.shape[0])

# --- 2. SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# --- 3. DATA AUGMENTATION ---
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# --- 4. MODEL ---
model = Sequential()

# Block 1
model.add(Conv2D(32, (5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

# Block 2
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

# Fully Connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# --- 5. TRAIN ---
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

print("Bắt đầu train...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# --- 6. SAVE MODEL ---
model.save("traffic_classifier.h5")
print("Đã lưu model!")

label_map = {
    0: "Giới hạn tốc độ 20 km/h",
    1: "Giới hạn tốc độ 30 km/h",
    2: "Giới hạn tốc độ 50 km/h",
    3: "Giới hạn tốc độ 60 km/h",
    4: "Giới hạn tốc độ 70 km/h",
    5: "Giới hạn tốc độ 80 km/h",
    6: "Hết giới hạn tốc độ 80 km/h",
    7: "Giới hạn tốc độ 100 km/h",
    8: "Giới hạn tốc độ 120 km/h",
    9: "Cấm vượt",
    10: "Cấm vượt xe trên 3.5 tấn",
    11: "Ưu tiên tại giao lộ",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại",
    15: "Cấm tất cả phương tiện",
    16: "Cấm xe trên 3.5 tấn",
    17: "Cấm vào",
    18: "Nguy hiểm chung",
    19: "Nguy hiểm: cua trái",
    20: "Nguy hiểm: cua phải",
    21: "Nguy hiểm: đường cong liên tiếp",
    22: "Đường gồ ghề",
    23: "Đường trơn trượt",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Đèn giao thông",
    27: "Người đi bộ",
    28: "Trẻ em qua đường",
    29: "Xe đạp qua đường",
    30: "Cảnh báo băng/tuyết",
    31: "Động vật hoang dã",
    32: "Hết mọi giới hạn tốc độ và cấm vượt",
    33: "Rẽ phải phía trước",
    34: "Rẽ trái phía trước",
    35: "Chỉ được đi thẳng",
    36: "Đi thẳng hoặc rẽ phải",
    37: "Đi thẳng hoặc rẽ trái",
    38: "Đi bên phải",
    39: "Đi bên trái",
    40: "Vòng xuyến",
    41: "Hết cấm vượt",
    42: "Hết cấm vượt xe trên 3.5 tấn"
}

with open("labels.json", "w") as f:
    json.dump(label_map, f)

print("Đã lưu labels.json!")

# --- 8. TEST CSV ---
print("Đang test với Test.csv...")

try:
    test_csv = pd.read_csv('Test.csv')
    labels_test = test_csv["ClassId"].values
    imgs_test = test_csv["Path"].values

    data_test = []

    for img_path in imgs_test:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((30, 30))
        data_test.append(np.array(img) / 255.0)

    X_test_final = np.array(data_test)

    pred = np.argmax(model.predict(X_test_final), axis=-1)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels_test, pred)

    print(f"Accuracy Test.csv: {acc*100:.2f}%")

except Exception as e:
    print("Lỗi test:", e)