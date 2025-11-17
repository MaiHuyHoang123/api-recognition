FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir ultralytics>=8.0.200 --no-deps
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000
CMD ["python3", "-m" ,"uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]