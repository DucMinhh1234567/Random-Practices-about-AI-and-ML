# Neural Network from scratch

Dự án luyện tập: mạng nơ-ron nhỏ (2 lớp) nhận diện chữ số viết tay kiểu MNIST, huấn luyện trong notebook (`better.ipynb`), trọng số lưu trong `model.pkl`.

Ứng dụng web Flask (`app.py`) đọc model đó, nhận 784 giá trị pixel qua API và trả về dự đoán cùng xác suất.

## Chạy thử

```bash
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tại `http://127.0.0.1:5000`.

Cần có file `model.pkl` (cùng thư mục với `app.py`). Nếu chưa có, chạy phần huấn luyện trong notebook để tạo file.

### Note:
Dataset MNIST có thể lấy trên kaggle

Các video tham khảo: 
- https://youtu.be/w8yWXqWQYmU?si=AJeMIM41OM7tFBu1
- https://youtu.be/cAkMcPfY_Ns?si=85JNXoH6CY2xh9PI
- https://youtu.be/aircAruvnKk?si=GxHQx-UJ5H0F4_WW
- https://youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&si=q_NMffo75gY3ICiX