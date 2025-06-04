# Start with a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies from your requirements.txt, and add gdown for Google Drive downloads
# Ensure gdown is not already in your requirements.txt to avoid conflicts
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gdown

# Clone the Wav2Lip repository (needed before creating subdirs for s3fd model)
# This will create a directory named Wav2Lip inside /app
RUN git clone https://github.com/Rudrabha/Wav2Lip.git /app/Wav2Lip

# Download the s3fd face detection model
# Create the target directory structure within the cloned Wav2Lip repo
RUN mkdir -p /app/Wav2Lip/face_detection/detection/sfd && \
    echo "Downloading s3fd face detection model..." && \
    wget -O /app/Wav2Lip/face_detection/detection/sfd/s3fd.pth https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

# Download the Wav2Lip checkpoint model from Google Drive
# Create the target directory for Wav2Lip checkpoint if it's not part of ./app/ copy
RUN mkdir -p /app/app/checkpoints && \
    echo "Downloading Wav2Lip GAN checkpoint model..." && \
    gdown --id 1_OvqStxNxLc7bXzlaVG5sz695p-FVfYY -O /app/app/checkpoints/wav2lip_gan.pth

# --- COPY YOUR APPLICATION CODE ---
# This assumes your local 'app' directory contains __init__.py (important!)
# and your other Python files (main.py, wav2lip_handler.py, etc.).
# This will copy the contents of your local ./app into /app/app/ in the container.
COPY ./app/ /app/app/
# If your local ./app directory does not have an __init__.py, the 'touch' command below would create it.
# However, it's best practice to have __init__.py in your local 'app' source directory.
# The line `RUN touch /app/app/__init__.py` from your previous Dockerfile is removed,
# assuming ./app/ already contains __init__.py. If not, re-add it or create it locally.

# Set environment variable for the Wav2Lip checkpoint path
ENV WAV2LIP_CHECKPOINT="/app/app/checkpoints/wav2lip_gan.pth"

# Add the cloned Wav2Lip directory and your main app directory to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/Wav2Lip:/app"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]