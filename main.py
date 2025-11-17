from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np
import requests
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ImageURLs(BaseModel):
    urls: list[str]

app = FastAPI(title="YOLO PPE Detection API")

model = YOLO("./models/best.pt")

# Tạo thread pool encode JPEG
executor = ThreadPoolExecutor(max_workers=8)

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

def encode_image(img):
    # Encode JPEG nhanh với chất lượng 80 (nhẹ hơn)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return encoded.tobytes() if ok else None

async def fetch_image(url, client):
    try:
        resp = await client.get(url, timeout=5)
        img_np = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except:
        return None
# -----------------------------
# API DETECT + UPLOAD
# -----------------------------
@app.post("/detect/")
async def detect_from_urls(data: ImageURLs):
    if not data.urls:
        return JSONResponse({"result_urls": []})
    print("tải ảnh------")
    # 1️⃣ Tải tất cả ảnh về (gom batch)
    async with httpx.AsyncClient() as client:
        tasks = [fetch_image(url, client) for url in data.urls]
        valid_images = [img for img in await asyncio.gather(*tasks) if img is not None]

    if not valid_images:
        return {"result_urls": []}
    print("predict------")
    # 2️⃣ YOLO predict 1 lần cho cả batch
    results = model(valid_images, conf=0.6, imgsz=960)

    # 3️⃣ Tạo annotated images
    annotated_images = [r.plot() for r in results]

    # 4️⃣ Encode JPEG song song bằng thread pool
    loop = asyncio.get_running_loop()
    encoded_images = await asyncio.gather(
        *[
            loop.run_in_executor(executor, encode_image, img)
            for img in annotated_images
        ]
    )

    # 4️⃣ Upload batch annotated images 1 lần
    uploaded_urls = upload_to_storage(encoded_images)

    return JSONResponse({
        "result_urls": uploaded_urls
    })