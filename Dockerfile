FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir f5-tts fastapi uvicorn python-multipart

COPY api/ ./api/
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
