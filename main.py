from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import requests

class ImageURLs(BaseModel):
    urls: list[str]

app = FastAPI(title="YOLO PPE Detection API")

model = YOLO("./models/best.pt")


# -----------------------------
# HÀM UPLOAD ẢNH LÊN API KHÁC
# -----------------------------
def upload_to_storage(image_bytes_list: list[bytes]) -> str:
    upload_api = "https://s3upload.hunonicpro.com/uploadAws.php"

    files = [
        ("uploaded_file[]", (f"img_{i}.jpg", img_bytes, "image/jpeg"))
        for i, img_bytes in enumerate(image_bytes_list)
    ]
    data = {
        "app_name": "wp_blugin_ai",
        "signature": "HUN_UPLOAD_IMAGE",
        "user_id": 437262,
        "is_image": 1
    }
    response = requests.post(upload_api, files=files, data=data, timeout=25)

    if response.status_code != 200:
        raise Exception("Upload image failed!")

    data = response.json()
    img_urls = []
    for image in data.get("data", []):
        img_urls.append(image.get("urlImage"))
    return img_urls


# -----------------------------
# API DETECT + UPLOAD
# -----------------------------
@app.post("/detect/")
async def detect_from_urls(data: ImageURLs):
    if not data.urls:
        return JSONResponse({"result_urls": []})
    
    # 1️⃣ Tải tất cả ảnh về (gom batch)
    images = []
    for url in data.urls:
        try:
            img_bytes = requests.get(url).content
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            images.append(img)
        except Exception as e:
            print("Error processing URL:", url, e)
            print("Error:", e)

    # Xử lý ảnh lỗi
    valid_images = [img for img in images if img is not None]
    if len(valid_images) == 0:
        return JSONResponse({"result_urls": []})

    # 2️⃣ YOLO predict 1 lần cho cả batch
    results = model(valid_images, verbose=False)

    # 3️⃣ Tạo annotated images
    annotated_list = []
    for result in results:
        annotated = result.plot()
        _, encoded = cv2.imencode(".jpg", annotated)
        annotated_list.append(encoded.tobytes())

    # 4️⃣ Upload batch annotated images 1 lần
    uploaded_urls = upload_to_storage(annotated_list)

    return JSONResponse({
        "result_urls": uploaded_urls
    })