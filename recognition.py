import cv2
import os
import random
from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO('./runs_v3/train/weights/best.pt')

# Mở webcam (0 = webcam mặc định, nếu không được thì thử 1 hoặc 2)
cap = cv2.VideoCapture(0)

# Kiểm tra webcam
if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Tạo thư mục lưu kết quả
os.makedirs('./output_PPE', exist_ok=True)

print("✅ Bắt đầu phát hiện đối tượng. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được khung hình từ webcam.")
        break

    # Dự đoán với YOLO
    results = model(frame, verbose=False)

    # Lấy frame gốc để vẽ
    annotated_frame = frame.copy()

    if results:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[class_id]

            # Màu sắc phân loại an toàn / nguy hiểm
            if label in ["no-helmet", "no-vest", "no-boot", "bare-arms"]:
                color = (0, 0, 255)  # Đỏ
            else:
                color = (0, 255, 0)  # Xanh

            # Vẽ khung và nhãn
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness=3)
            text = f"{label} {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10

            cv2.rectangle(annotated_frame,
                          (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5),
                          color, -1)
            cv2.putText(annotated_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Hiển thị kết quả
    cv2.imshow("YOLO PPE Detection", annotated_frame)

    # Nhấn 's' để lưu khung hình lại
    if cv2.waitKey(1) & 0xFF == ord('s'):
        random_number = random.randint(0, 100000)
        output_path = os.path.join('./output_PPE', f'frame_{random_number}.jpg')
        cv2.imwrite(output_path, annotated_frame)
        print(f"💾 Đã lưu khung hình: {output_path}")

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("✅ Đã thoát.")
