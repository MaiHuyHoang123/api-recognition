import os
import random
import cv2
import time
from ultralytics import YOLO  # Giả định YOLO từ thư viện ultralytics
import psutil

# Load mô hình đã huấn luyện
model = YOLO('./runs_v3/train/weights/best.pt')  # 'best.pt' là đường dẫn đến mô hình đã được train

# Đọc hình ảnh đầu vào
img = cv2.imread('./image/images.jpg')

# Thực hiện dự đoán trên hình ảnh
results = model(img)

# Lưu kết quả dự đoán vào thư mục output_PPE
os.makedirs('./output_PPE', exist_ok=True)

if results:
        # Đo tài nguyên trước khi chạy
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used / (1024 ** 3)
    start = time.time()

    annotated_frame = results[0].plot()
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        confidence = box.conf.item()
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        label = model.names[class_id]

        # Màu sắc cho nhãn an toàn (màu xanh) và nhãn nguy hiểm (màu đỏ)
        if label == "no-helmet" or label == "no-vest" or label == "no-boot" or label == "bare-arms":
            color = (0, 0, 255)  # Màu đỏ
        else:
            color = (0, 255, 0)  # Màu xanh

        # Vẽ khung với độ dày cố định
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)

        # Cài đặt văn bản và kích thước văn bản cố định
        text = f"{label} {confidence:.2f}"
        text_color = (0, 0, 0)  # Màu đen
        font_scale = 0.6
        font_thickness = 1

        # Tính toán vị trí hiển thị của nhãn
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10

        # Vẽ hình chữ nhật làm nền nhãn để tăng độ tương phản
        cv2.rectangle(img, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), color, -1)
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        end = time.time()

        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().used / (1024 ** 3)

print(f"CPU sử dụng trung bình: ~{(cpu_before + cpu_after)/2:.2f}%")
print(f"RAM tăng: {mem_after - mem_before:.2f} GB")
print(f"Thời gian chạy: {end - start:.2f} giây")
# Tạo tên file ngẫu nhiên và lưu hình ảnh kết quả
random_number = random.randint(0, 100000)
output_path = os.path.join('./output_PPE', f'predicted_image_{random_number}.jpg')
cv2.imwrite(output_path, img)
print(f"Hình ảnh đã được lưu tại: {output_path}")
