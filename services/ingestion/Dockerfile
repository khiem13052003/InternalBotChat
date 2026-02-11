FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ingestion

COPY . .
RUN pip install --upgrade pip
RUN pip install torch --no-cache-dir
RUN pip install numpy
RUN pip install --no-cache-dir -r requirements.txt 
CMD ["python", "app.py"]
