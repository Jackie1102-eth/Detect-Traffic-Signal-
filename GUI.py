import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from PIL import Image, ImageTk
import numpy as np
import os
import json

#  LOAD LABELS                       
def load_labels():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

#  LOAD MODEL 
def load_model_once():
    try:
        from tensorflow.keras.models import load_model
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_classifier.h5")
        if not os.path.exists(path):
            return None, f"Không tìm thấy:\n{path}"
        return load_model(path), None
    except Exception as e:
        return None, str(e)

#  PREDICT — trả về top 3                                              
def predict_top3(model, pil_image):
    img = pil_image.convert("RGB").resize((30, 30))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]
    return [(int(i), float(preds[i])) for i in top3]

#  GIAO DIỆN                                                           #
class App(tk.Tk):

    C_BG     = "#F4F4F1"
    C_HEADER = "#1A1A1A"
    C_WHITE  = "#FFFFFF"
    C_BORDER = "#DDDDDD"
    C_GREEN  = "#1D9E75"
    C_AMBER  = "#B8860B"
    C_RED    = "#D85A30"
    C_GRAY   = "#888888"
    C_STATUS = "#E6E6E1"

    UNKNOWN_THRESHOLD = 0.30   # < 30% → coi là unknown

    def __init__(self):
        super().__init__()
        self.title("Traffic Sign AI")
        self.geometry("540x660")
        self.resizable(False, False)
        self.configure(bg=self.C_BG)

        self.model       = None
        self.class_names = load_labels()
        self.current_pil = None
        self._photo      = None

        self._build_ui()
        self._load_model_async()

    #  BUILD UI                                                          


    def _build_ui(self):
        # Header
        bar = tk.Frame(self, bg=self.C_HEADER, height=52)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="🚦  Traffic Sign Detector",
                 font=("Helvetica", 15, "bold"),
                 fg="white", bg=self.C_HEADER).pack(side=tk.LEFT, padx=18, pady=12)
        self.model_lbl = tk.Label(bar, text="⏳  Đang tải model...",
                                   font=("Helvetica", 9),
                                   fg="#AAAAAA", bg=self.C_HEADER)
        self.model_lbl.pack(side=tk.RIGHT, padx=18)

        # Khung ảnh
        img_wrap = tk.Frame(self, bg=self.C_BG)
        img_wrap.pack(padx=30, pady=(20, 0))
        img_frame = tk.Frame(img_wrap, bg=self.C_WHITE, width=480, height=300,
                              highlightbackground=self.C_BORDER, highlightthickness=1)
        img_frame.pack()
        img_frame.pack_propagate(False)
        self.img_label = tk.Label(img_frame, bg=self.C_WHITE,
                                   text="Chưa có ảnh\nNhấn  Mở ảnh  để bắt đầu",
                                   font=("Helvetica", 11), fg="#BBBBBB")
        self.img_label.pack(expand=True)

        # Tên biển (top 1)
        self.sign_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.sign_var,
                 font=("Helvetica", 16, "bold"),
                 fg=self.C_HEADER, bg=self.C_BG,
                 wraplength=480, justify=tk.CENTER).pack(pady=(16, 0))

        # Độ tự tin top 1
        self.conf_var = tk.StringVar(value="")
        self.conf_lbl = tk.Label(self, textvariable=self.conf_var,
                                  font=("Helvetica", 11),
                                  fg=self.C_GRAY, bg=self.C_BG)
        self.conf_lbl.pack(pady=(3, 0))

        # Separator
        tk.Frame(self, bg=self.C_BORDER, height=1).pack(fill=tk.X, padx=30, pady=(12, 0))

        # Top 3 box
        top3_wrap = tk.Frame(self, bg=self.C_BG)
        top3_wrap.pack(fill=tk.X, padx=30, pady=(8, 0))
        tk.Label(top3_wrap, text="Top 3 dự đoán",
                 font=("Helvetica", 9), fg=self.C_GRAY,
                 bg=self.C_BG).pack(anchor=tk.W)

        self.top3_frame = tk.Frame(top3_wrap, bg=self.C_BG)
        self.top3_frame.pack(fill=tk.X, pady=(4, 0))

        self.top3_rows = []
        for _ in range(3):
            row = tk.Frame(self.top3_frame, bg=self.C_BG)
            row.pack(fill=tk.X, pady=2)
            rank_lbl = tk.Label(row, text="", font=("Helvetica", 10, "bold"),
                                 width=2, bg=self.C_BG, fg=self.C_GRAY)
            rank_lbl.pack(side=tk.LEFT)
            name_lbl = tk.Label(row, text="", font=("Helvetica", 10),
                                 bg=self.C_BG, fg=self.C_HEADER, anchor=tk.W)
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            conf_lbl = tk.Label(row, text="", font=("Helvetica", 10, "bold"),
                                 width=8, bg=self.C_BG, fg=self.C_GRAY, anchor=tk.E)
            conf_lbl.pack(side=tk.RIGHT)
            self.top3_rows.append((rank_lbl, name_lbl, conf_lbl))

        # Progress bar
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=480)
        self.progress.pack(pady=(12, 0))

        # Nút
        btn_row = tk.Frame(self, bg=self.C_BG)
        btn_row.pack(pady=14)

        self.open_btn = tk.Button(btn_row, text="📂  Mở ảnh",
                                   font=("Helvetica", 11),
                                   bg=self.C_HEADER, fg="white",
                                   activebackground="#333",
                                   relief=tk.FLAT, padx=18, pady=9,
                                   cursor="hand2", command=self._open_image)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.detect_btn = tk.Button(btn_row, text="🔍  Nhận diện",
                                     font=("Helvetica", 11),
                                     bg=self.C_GREEN, fg="white",
                                     activebackground="#0F6E56",
                                     relief=tk.FLAT, padx=18, pady=9,
                                     cursor="hand2", state=tk.DISABLED,
                                     command=self._detect_async)
        self.detect_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.reset_btn = tk.Button(btn_row, text="🗑  Xoá",
                                    font=("Helvetica", 11),
                                    bg="#E8E8E3", fg="#444",
                                    activebackground="#D8D8D3",
                                    relief=tk.FLAT, padx=18, pady=9,
                                    cursor="hand2", state=tk.DISABLED,
                                    command=self._reset)
        self.reset_btn.pack(side=tk.LEFT)

        # Status bar
        self.status_lbl = tk.Label(self, text="  Sẵn sàng",
                                    font=("Helvetica", 9),
                                    bg=self.C_STATUS, fg="#666", anchor=tk.W)
        self.status_lbl.pack(fill=tk.X, side=tk.BOTTOM, ipady=4)


    #  LOGIC                                                             


    def _load_model_async(self):
        def task():
            m, err = load_model_once()
            self.after(0, self._on_model_loaded, m, err)
        threading.Thread(target=task, daemon=True).start()

    def _on_model_loaded(self, m, err):
        if m:
            self.model = m
            self.model_lbl.config(text="✅  Model sẵn sàng", fg=self.C_GREEN)
            self._set_status("Model đã tải — hãy mở ảnh biển báo.")
        else:
            self.model_lbl.config(text="❌  Lỗi model", fg=self.C_RED)
            self._set_status(f"Lỗi: {err}", error=True)
            messagebox.showerror("Lỗi load model",
                                 f"{err}\n\nĐặt file traffic_classifier.h5 cùng thư mục với script.")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh biển báo",
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.ppm"), ("Tất cả", "*.*")]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            self.current_pil = pil
            self._show_image(pil)
            self._clear_result()
            self.detect_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
            self._set_status(f"Đã tải: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", str(e))

    def _show_image(self, pil_img):
        disp = pil_img.copy()
        disp.thumbnail((480, 300), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)
        self.img_label.config(image=self._photo, text="")

    def _detect_async(self):
        if not self.model or not self.current_pil:
            return
        self._toggle_buttons(True)
        self.progress.start(12)
        self._clear_result()
        self._set_status("Đang nhận diện...")

        def task():
            try:
                results = predict_top3(self.model, self.current_pil)
                self.after(0, self._show_result, results)
            except Exception as e:
                self.after(0, self._on_error, str(e))

        threading.Thread(target=task, daemon=True).start()

    def _show_result(self, results):
        self.progress.stop()
        self._toggle_buttons(False)

        if not results:
            self._set_status("Không có kết quả.", error=True)
            return

        cid, conf = results[0]
        pct = conf * 100

        medals = ["🥇", "🥈", "🥉"]

        # ── UNKNOWN: confidence quá thấp 
        if conf < self.UNKNOWN_THRESHOLD:
            self.sign_var.set("❓  Không nhận ra biển báo")
            self.conf_var.set(
                f"Độ tự tin quá thấp ({pct:.1f}%) — ảnh không phải biển báo giao thông"
            )
            self.conf_lbl.config(fg=self.C_RED)

            for idx, (rank_lbl, name_lbl, conf_lbl) in enumerate(self.top3_rows):
                if idx < len(results):
                    c, p = results[idx]
                    rank_lbl.config(text=medals[idx])
                    name_lbl.config(text=self._get_name(c))
                    conf_lbl.config(text=f"{p*100:.1f}%", fg=self.C_RED)
                else:
                    rank_lbl.config(text="")
                    name_lbl.config(text="")
                    conf_lbl.config(text="")

            self._set_status("⚠️  Ảnh không được nhận diện là biển báo.", error=True)
            return
        # ── END UNKNOWN ───────────────────────────────────────────────

        # Top 1 bình thường
        name = self._get_name(cid)
        self.sign_var.set(name)

        if pct >= 70:
            color, icon = self.C_GREEN, "✅"
        elif pct >= 40:
            color, icon = self.C_AMBER, "⚠️"
        else:
            color, icon = self.C_RED, "❌"

        self.conf_var.set(f"{icon}  Độ tự tin: {pct:.1f}%")
        self.conf_lbl.config(fg=color)

        # Top 3 rows
        for idx, (rank_lbl, name_lbl, conf_lbl) in enumerate(self.top3_rows):
            if idx < len(results):
                c, p = results[idx]
                rank_lbl.config(text=medals[idx])
                name_lbl.config(text=self._get_name(c))
                conf_lbl.config(text=f"{p*100:.1f}%",
                                 fg=self.C_GREEN if p >= 0.7 else
                                    self.C_AMBER  if p >= 0.4 else self.C_RED)
            else:
                rank_lbl.config(text="")
                name_lbl.config(text="")
                conf_lbl.config(text="")

        self._set_status("Nhận diện hoàn tất!")

    def _on_error(self, msg):
        self.progress.stop()
        self._toggle_buttons(False)
        self._set_status(f"Lỗi: {msg}", error=True)
        messagebox.showerror("Lỗi nhận diện", msg)

    def _reset(self):
        self.current_pil = None
        self._photo = None
        self.img_label.config(image="",
                               text="Chưa có ảnh\nNhấn  Mở ảnh  để bắt đầu")
        self._clear_result()
        self.detect_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self._set_status("Đã xoá.")


    #  HELPERS                                                           


    def _get_name(self, class_id: int) -> str:
        if self.class_names and class_id in self.class_names:
            return self.class_names[class_id]
        return f"Class {class_id}"

    def _clear_result(self):
        self.sign_var.set("")
        self.conf_var.set("")
        self.conf_lbl.config(fg=self.C_GRAY)
        for rank_lbl, name_lbl, conf_lbl in self.top3_rows:
            rank_lbl.config(text="")
            name_lbl.config(text="")
            conf_lbl.config(text="")

    def _toggle_buttons(self, detecting: bool):
        s = tk.DISABLED if detecting else tk.NORMAL
        self.open_btn.config(state=s)
        self.detect_btn.config(state=s)
        self.reset_btn.config(state=s)

    def _set_status(self, msg, error=False):
        self.status_lbl.config(text=f"  {msg}",
                                fg=self.C_RED if error else "#555")

if __name__ == "__main__":
    app = App()
    app.mainloop()