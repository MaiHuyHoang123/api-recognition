from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI(title="YOLO PPE Detection API")

# Load model
model = YOLO("runs_v3/train/weights/best.pt")

@app.get("/")
def root():
    return {"message": "YOLO PPE Detection API is running 🚀"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Đọc file ảnh từ request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Chạy dự đoán YOLO
    results = model(img, verbose=False)
    boxes_info = []
    output_path = os.path.join('./output_PPE', f'predicted_image.jpg')
    if results:
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = model.names[class_id]
            xyxy = list(map(int, box.xyxy[0].tolist()))
            annotated_frame = results[0].plot()
            cv2.imwrite(output_path, annotated_frame)
            boxes_info.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": xyxy
            })

    return FileResponse(output_path, media_type="image/jpeg")
