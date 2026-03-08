FROM python:3.10-slim

WORKDIR /tts

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . /tts

CMD ["sh", "-c", "cd app; uvicorn main:server --host 0.0.0.0 --port 4313 --workers 1"]
