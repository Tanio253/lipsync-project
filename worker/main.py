# worker/main.py (Corrected and Simplified)
import base64
import json
import logging
import os
import time

import cv2
import numpy as np
import redis
import torch

import face_detection
from models import Wav2Lip
from wav2lip_inference_logic import load_wav2lip_model
from streaming_processor import Wav2LipStreamingSession, IMG_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379
DEVICE = os.environ.get("DEVICE", 'cuda' if torch.cuda.is_available() else 'cpu')
WAV2LIP_CHECKPOINT_PATH = os.environ.get("WAV2LIP_CHECKPOINT", "worker/checkpoints/wav2lip_gan.pth")
WORKER_QUEUE_NAME = "worker_queue"

WAV2LIP_MODEL: Wav2Lip = None
FACE_DETECTOR: face_detection.FaceAlignment = None
ACTIVE_SESSIONS = {}

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

def load_models():
    global WAV2LIP_MODEL, FACE_DETECTOR
    logger.info(f"Loading models onto device: {DEVICE}")
    try:
        WAV2LIP_MODEL = load_wav2lip_model(WAV2LIP_CHECKPOINT_PATH).to(DEVICE).eval()
        FACE_DETECTOR = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=DEVICE)
        logger.info("Wav2Lip and FaceAlignment models loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load models on startup: {e}", exc_info=True)
        raise

def create_session(image_b64: str) -> tuple:
    image_bytes = base64.b64decode(image_b64)
    original_face_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if original_face_cv2 is None: raise ValueError("Could not decode face image.")
    detected_boxes = FACE_DETECTOR.get_detections_for_batch(np.expand_dims(original_face_cv2, axis=0))
    if not detected_boxes or detected_boxes[0] is None: raise ValueError("Face not detected.")
    rect = detected_boxes[0]
    pads = [0, 10, 0, 0]
    y1, y2, x1, x2 = (
        max(0, int(rect[1] - pads[0])), min(original_face_cv2.shape[0], int(rect[3] + pads[1])),
        max(0, int(rect[0] - pads[2])), min(original_face_cv2.shape[1], int(rect[2] + pads[3]))
    )
    face_crop_cv2 = original_face_cv2[y1:y2, x1:x2]
    face_resized_cv2 = cv2.resize(face_crop_cv2, (IMG_SIZE, IMG_SIZE))
    img_masked = face_resized_cv2.copy()
    img_masked[IMG_SIZE // 2:, :] = 0
    processed_face_hwc = np.concatenate((img_masked, face_resized_cv2), axis=2) / 255.0
    face_tensor_template = torch.FloatTensor(np.transpose(processed_face_hwc, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    return Wav2LipStreamingSession(face_tensor_template, (y1, y2, x1, x2), original_face_cv2.copy())

def main_poll_loop():
    logger.info("Worker started. Waiting for jobs from Redis...")
    while True:
        try:
            _, job_raw = redis_client.blpop(WORKER_QUEUE_NAME)
            job = json.loads(job_raw)
            job_type = job.get("type")
            session_id = job.get("session_id")
            client_id = job.get("client_id") # Present in all relevant jobs now

            if not client_id and job_type != "cleanup_session":
                 logger.warning(f"Received job of type {job_type} without a client_id. Skipping.")
                 continue
            
            results_channel = f"results:{client_id}"

            if job_type == "start_session":
                try:
                    session = create_session(job["image_b64"])
                    ACTIVE_SESSIONS[session_id] = session
                    redis_client.publish(results_channel, json.dumps({"status": "Session started, ready for audio."}))
                    logger.info(f"Successfully created session {session_id} for client {client_id}")
                except Exception as e:
                    logger.error(f"Failed to create session {session_id}: {e}", exc_info=True)
                    redis_client.publish(results_channel, json.dumps({"error": f"Session start error: {str(e)}"}))

            elif job_type == "process_audio":
                if session_id not in ACTIVE_SESSIONS: continue
                
                session = ACTIVE_SESSIONS[session_id]
                session.add_audio_chunk(base64.b64decode(job["audio_chunk_b64"]))
                mel_chunks = session.get_available_mel_chunks()
                if not mel_chunks: continue

                mel_tensors = torch.FloatTensor(np.array(mel_chunks)).unsqueeze(1).to(DEVICE)
                face_batch = session.face_tensor_template.repeat(len(mel_tensors), 1, 1, 1)

                with torch.no_grad():
                    generated_faces_tensor = WAV2LIP_MODEL(mel_tensors, face_batch)
                
                generated_faces_np = generated_faces_tensor.cpu().numpy()

                for face_np in generated_faces_np:
                    final_frame = session.composite_frame(face_np)
                    _, buffer = cv2.imencode('.jpg', final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    redis_client.publish(results_channel, json.dumps({"type": "video_frame", "frame_base64": frame_b64}))

            elif job_type == "cleanup_session":
                if session_id in ACTIVE_SESSIONS:
                    del ACTIVE_SESSIONS[session_id]
                    logger.info(f"Cleaned up session: {session_id}")

        except Exception as e:
            logger.error(f"An error occurred in the main poll loop: {e}", exc_info=True)
            time.sleep(1)

if __name__ == '__main__':
    load_models()
    main_poll_loop()