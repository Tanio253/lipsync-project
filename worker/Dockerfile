FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

# Copy the rest of your application code
COPY . .

# Download the external Wav2Lip code and models.
RUN git clone https://github.com/Rudrabha/Wav2Lip.git

RUN touch /app/Wav2Lip/__init__.py

RUN mkdir -p /app/Wav2Lip/face_detection/detection/sfd && \
    echo "Downloading s3fd face detection model..." && \
    wget -O /app/Wav2Lip/face_detection/detection/sfd/s3fd.pth https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

RUN mkdir -p /app/checkpoints && \
    echo "Downloading Wav2Lip GAN checkpoint model..." && \
    gdown --id 1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY -O /app/checkpoints/wav2lip_gan.pth

# Set environment variables
ENV WAV2LIP_CHECKPOINT="/app/checkpoints/wav2lip_gan.pth"
ENV PYTHONPATH="${PYTHONPATH}:/app/Wav2Lip"



EXPOSE 8000

CMD ["python", "main.py"]