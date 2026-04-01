import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import os

class TrafficSignApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ Thống Nhận Diện Biển Báo Giao Thông - NCKH")
        self.window.geometry("1000x700")
        self.window.configure(bg="#1e1e2f") # Giao diện tối hiện đại

        # --- 1. TỪ ĐIỂN TIẾNG VIỆT (43 LỚP GTSRB) ---
        self.classes = { 
            0:'Hạn chế tốc độ (20km/h)', 1:'Hạn chế tốc độ (30km/h)', 2:'Hạn chế tốc độ (50km/h)', 
            3:'Hạn chế tốc độ (60km/h)', 4:'Hạn chế tốc độ (70km/h)', 5:'Hạn chế tốc độ (80km/h)', 
            6:'Hết hạn chế tốc độ (80km/h)', 7:'Hạn chế tốc độ (100km/h)', 8:'Hạn chế tốc độ (120km/h)', 
            9:'Cấm vượt', 10:'Cấm xe tải vượt', 11:'Giao nhau với đường không ưu tiên', 
            12:'Đường ưu tiên', 13:'Nhường đường', 14:'Dừng lại (Stop)', 15:'Cấm tất cả phương tiện', 
            16:'Cấm xe tải', 17:'Cấm đi ngược chiều', 18:'Nguy hiểm khác', 19:'Chỗ ngoặt nguy hiểm bên trái', 
            20:'Chỗ ngoặt nguy hiểm bên phải', 21:'Nhiều chỗ ngoặt nguy hiểm liên tiếp', 22:'Đường lồi lõm', 
            23:'Đường trơn trượt', 24:'Đường bị hẹp bên phải', 25:'Công trường đang thi công', 
            26:'Tín hiệu đèn giao thông', 27:'Người đi bộ cắt ngang', 28:'Trẻ em đi ngang đường', 
            29:'Người đi xe đạp cắt ngang', 30:'Cẩn thận băng tuyết', 31:'Động vật hoang dã đi ngang', 
            32:'Hết tất cả lệnh cấm', 33:'Chỉ dẫn rẽ phải phía trước', 34:'Chỉ dẫn rẽ trái phía trước', 
            35:'Chỉ dẫn đi thẳng', 36:'Chỉ dẫn đi thẳng hoặc rẽ phải', 37:'Chỉ dẫn đi thẳng hoặc rẽ trái', 
            38:'Đi bên phải để tránh vật cản', 39:'Đi bên trái để tránh vật cản', 40:'Vòng xuyến', 
            41:'Hết cấm vượt', 42:'Hết cấm xe tải vượt' 
        }

        # --- 2. LOAD MÔ HÌNH AI ---
        try:
            # Hãy đảm bảo file này đã được tạo ra từ file train
            self.model = load_model('traffic_classifier.h5')
            print("Đã nạp mô hình thành công!")
        except:
            messagebox.showerror("Lỗi", "Không tìm thấy file 'traffic_classifier.h5'. Hãy chạy train trước!")
            self.window.destroy()

        # --- 3. GIAO DIỆN (UI) ---
        # Tiêu đề
        self.lbl_header = tk.Label(window, text="NHẬN DIỆN BIỂN BÁO GIAO THÔNG REAL-TIME", 
                                   font=("Arial", 22, "bold"), bg="#1e1e2f", fg="#00df9a", pady=20)
        self.lbl_header.pack()

        # Khung chứa Video và Kết quả
        self.container = tk.Frame(window, bg="#1e1e2f")
        self.container.pack(expand=True)

        # Cột trái: Webcam
        self.video_label = tk.Label(self.container, bg="black", bd=2, relief="solid")
        self.video_label.grid(row=0, column=0, padx=20)

        # Cột phải: Thông tin nhận diện
        self.info_frame = tk.Frame(self.container, bg="#2d2d44", padx=30, pady=30, bd=1, relief="ridge")
        self.info_frame.grid(row=0, column=1, sticky="nsew")

        tk.Label(self.info_frame, text="KẾT QUẢ PHÂN LOẠI", font=("Arial", 14, "bold"), bg="#2d2d44", fg="white").pack()
        
        self.res_text = tk.Label(self.info_frame, text="Chưa có dữ liệu", font=("Arial", 16, "italic"), 
                                 bg="#2d2d44", fg="#f1c40f", wraplength=200, pady=20)
        self.res_text.pack()

        self.conf_text = tk.Label(self.info_frame, text="Độ tin cậy: 0%", font=("Arial", 12), bg="#2d2d44", fg="#bdc3c7")
        self.conf_text.pack()

        # Nút bấm
        self.btn_quit = tk.Button(window, text="THOÁT ỨNG DỤNG", command=self.cleanup, 
                                  bg="#ff4b2b", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10)
        self.btn_quit.pack(pady=30)

        # --- 4. XỬ LÝ VIDEO ---
        self.cap = cv2.VideoCapture(0)
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Tiền xử lý ảnh cho Model (phải khớp với lúc Train: 30x30)
            img = cv2.resize(frame, (30, 30))
            img = np.expand_dims(img, axis=0)
            img = np.array(img)

            # 2. Dự đoán
            pred_prob = self.model.predict(img, verbose=0)
            class_id = np.argmax(pred_prob)
            confidence = np.max(pred_prob) * 100

            # 3. Cập nhật chữ Tiếng Việt nếu độ tin cậy > 80% (để tránh nhảy chữ lung tung)
            if confidence > 80:
                self.res_text.configure(text=self.classes[class_id], fg="#00df9a")
                self.conf_text.configure(text=f"Độ tin cậy: {confidence:.2f}%")
            else:
                self.res_text.configure(text="Đang nhận diện...", fg="#bdc3c7")
                self.conf_text.configure(text="Độ tin cậy: < 80%")

            # 4. Hiển thị Video lên GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        self.window.after(10, self.update)

    def cleanup(self):
        self.cap.release()
        self.window.destroy()

# Khởi chạy
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignApp(root)
    root.mainloop()