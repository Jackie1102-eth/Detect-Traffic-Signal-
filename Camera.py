"""
Nhận Diện Biển Báo Giao Thông Bằng Camera
==========================================
Cách chạy:
    pip install tensorflow pillow opencv-python numpy
    python camera_detect.py

Yêu cầu: file traffic_classifier.h5 nằm cùng thư mục với file này.

Phím tắt khi cửa sổ camera đang mở:
    Q  —  Thoát
    S  —  Chụp và lưu ảnh hiện tại
"""

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------ #
#  Cấu hình                                                            #
# ------------------------------------------------------------------ #

MODEL_PATH  = "traffic_classifier.h5"
IMG_SIZE    = 30          # Kích thước model yêu cầu
CONF_THRESH = 0.70        # Chỉ hiển thị kết quả nếu độ tự tin >= 70%
CAM_INDEX   = 0           # 0 = webcam mặc định, đổi thành 1/2 nếu có nhiều camera
ROI_SIZE    = 200         # Kích thước vùng quét (pixel) ở giữa màn hình

CLASS_NAMES = {
    0: "Gioi han toc do 20km/h",   1: "Gioi han toc do 30km/h",
    2: "Gioi han toc do 50km/h",   3: "Gioi han toc do 60km/h",
    4: "Gioi han toc do 70km/h",   5: "Gioi han toc do 80km/h",
    6: "Het gioi han 80km/h",      7: "Gioi han toc do 100km/h",
    8: "Gioi han toc do 120km/h",  9: "Cam vuot",
    10: "Cam vuot xe >3.5T",       11: "Uu tien nga tu",
    12: "Duong uu tien",           13: "Nhuong duong",
    14: "Dung lai (STOP)",         15: "Cam xe co",
    16: "Cam xe tai >3.5T",        17: "Cam vao",
    18: "Chu y nguy hiem",         19: "Cua trai nguy hiem",
    20: "Cua phai nguy hiem",      21: "Duong cong lien tiep",
    22: "Duong go ghe",            23: "Duong tron truot",
    24: "Duong hep ben phai",      25: "Cong trinh duong bo",
    26: "Co tin hieu den",         27: "Nguoi di bo",
    28: "Tre em qua duong",        29: "Xe dap qua duong",
    30: "Chu y bang/tuyet",        31: "Thu hoang qua duong",
    32: "Het moi gioi han",        33: "Re phai phia truoc",
    34: "Re trai phia truoc",      35: "Di thang",
    36: "Di thang hoac re phai",   37: "Di thang hoac re trai",
    38: "Giu ben phai",            39: "Giu ben trai",
    40: "Vong xuyen bat buoc",     41: "Het cam vuot",
    42: "Het cam vuot xe >3.5T",
}

# ------------------------------------------------------------------ #
#  Hàm dự đoán                                                         #
# ------------------------------------------------------------------ #

def predict_frame(model, roi_bgr):
    """
    Nhận vùng ảnh BGR (numpy array), trả về (class_id, confidence).
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(roi_resized / 255.0, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    return class_id, confidence


def draw_overlay(frame, class_id, confidence, roi_x1, roi_y1, roi_x2, roi_y2):
    """Vẽ khung ROI và kết quả lên frame."""
    h, w = frame.shape[:2]

    # Màu khung ROI theo độ tự tin
    if confidence >= CONF_THRESH:
        box_color = (0, 200, 80)    # xanh lá — tin cậy
        label = f"{CLASS_NAMES.get(class_id, '???')}  {confidence*100:.1f}%"
    else:
        box_color = (0, 150, 255)   # cam — chưa chắc
        label = f"Dang phan tich...  {confidence*100:.1f}%"

    # Vẽ khung ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, 2)

    # Góc khung (trang trí)
    corner = 20
    for (cx, cy, dx, dy) in [
        (roi_x1, roi_y1,  1,  1),
        (roi_x2, roi_y1, -1,  1),
        (roi_x1, roi_y2,  1, -1),
        (roi_x2, roi_y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), box_color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), box_color, 3)

    # Nền nhãn
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = roi_x1
    label_y = roi_y1 - 10 if roi_y1 - 10 > th + 6 else roi_y2 + th + 10

    cv2.rectangle(frame,
                  (label_x, label_y - th - 6),
                  (label_x + tw + 10, label_y + baseline),
                  box_color, -1)
    cv2.putText(frame, label,
                (label_x + 5, label_y - 3),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Hướng dẫn góc trái dưới
    hint = "Q: Thoat  |  S: Chup anh"
    cv2.putText(frame, hint,
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ------------------------------------------------------------------ #
#  Vòng lặp chính                                                      #
# ------------------------------------------------------------------ #

def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[LOI] Khong tim thay file model: {MODEL_PATH}")
        print("      Hay dat traffic_classifier.h5 cung thu muc voi script nay.")
        return

    print("Dang tai model...")
    model = load_model(MODEL_PATH)
    print("Model da san sang!")

    # Mở camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo camera (index={CAM_INDEX}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera dang mo. Nhan Q de thoat, S de chup anh.")
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LOI] Khong doc duoc frame tu camera.")
            break

        h, w = frame.shape[:2]

        # Tính vùng ROI ở trung tâm frame
        cx, cy  = w // 2, h // 2
        half    = ROI_SIZE // 2
        roi_x1  = max(cx - half, 0)
        roi_y1  = max(cy - half, 0)
        roi_x2  = min(cx + half, w)
        roi_y2  = min(cy + half, h)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Dự đoán
        class_id, confidence = predict_frame(model, roi)

        # Vẽ overlay
        frame = draw_overlay(frame, class_id, confidence,
                             roi_x1, roi_y1, roi_x2, roi_y2)

        cv2.imshow("Nhan Dien Bien Bao - Q de thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_count += 1
            filename = f"capture_{save_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Da luu: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("Da dong camera.")


if __name__ == "__main__":
    main()