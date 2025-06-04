# app/streaming_processor.py
import cv2
import numpy as np
import torch
import logging

# Assuming Wav2Lip's utility modules are accessible
# These modules (audio.py, face_detection.py) are part of the Wav2Lip repository
# Ensure they are in your project structure, e.g., in the 'app' directory or PYTHONPATH
try:
    import audio # From Wav2Lip (for melspectrogram, STFT parameters)
    import face_detection # From Wav2Lip (for FaceAlignment)
except ImportError as e:
    logging.error(f"Failed to import Wav2Lip utility modules (audio, face_detection): {e}. Ensure they are in the PYTHONPATH.")
    raise

logger = logging.getLogger(__name__)

# --- Audio Processing Parameters (must align with Wav2Lip's audio.py and model training) ---
SAMPLE_RATE = 16000  # Hz
# Parameters from Wav2Lip's audio.py (or common defaults if not explicitly stated there)
# These are crucial for audio.melspectrogram to work as expected by the model.
# You might need to inspect Wav2Lip's audio.py to confirm these if issues arise.
N_FFT = 800       # Typical value from Wav2Lip's hparams or audio.py
HOP_LENGTH = 200  # Typical value from Wav2Lip's hparams or audio.py (samples per mel frame column)
FPS = 25          # Target video frames per second

MEL_STEP_SIZE = 16 # Width of mel spectrogram chunk Wav2Lip typically expects
IMG_SIZE = 96      # Default image size for Wav2Lip input

# Audio samples per video frame at the target FPS
SAMPLES_PER_VIDEO_FRAME = int(SAMPLE_RATE / FPS) # e.g., 16000 / 25 = 640 samples

# Audio samples needed to generate a mel spectrogram window of MEL_STEP_SIZE columns.
# This ensures enough context for the STFT to produce the required mel frames.
# Formula: (number_of_mel_frames - 1) * hop_length + n_fft
SAMPLES_FOR_MEL_WINDOW = (MEL_STEP_SIZE - 1) * HOP_LENGTH + N_FFT # e.g., (16-1)*200 + 800 = 3800 samples

class Wav2LipStreamingSession:
    def __init__(self, model_instance, face_image_bytes: bytes, device: str = 'cpu'):
        self.model = model_instance.to(device) # Ensure model is on the correct device
        self.device = device
        self.audio_buffer = np.array([], dtype=np.int16) # Buffer for S16LE PCM audio data

        logger.info(f"Initializing Wav2LipStreamingSession on device: {self.device}")

        # 1. Decode and store original face image (used for compositing)
        original_face_cv2 = cv2.imdecode(np.frombuffer(face_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if original_face_cv2 is None:
            logger.error("Could not decode face image from bytes.")
            raise ValueError("Could not decode face image.")
        self.original_face_cv2_for_compositing = original_face_cv2.copy()
        logger.info(f"Original face image loaded: {self.original_face_cv2_for_compositing.shape}")


        # 2. Perform Face Detection and Cropping (Adapted from Wav2Lip's face_detection logic)
        # This uses the FaceAlignment class from Wav2Lip's face_detection.py
        # Pads for face detection (top, bottom, left, right) - can be tuned
        pads = [0, 10, 0, 0] # Default from original Wav2Lip scripts
        face_det_batch_size = 1 # For single image detection

        try:
            # Initialize face detector (ideally, this could be global if resource-heavy)
            # For simplicity in session, initialize here. Ensure face_detection.py is in path.
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                     flip_input=False, device=self.device)
        except Exception as e:
            logger.error(f"Failed to initialize FaceAlignment: {e}. Check face_detection.py and CUDA/CPU setup.")
            raise

        # Detect face in the provided image
        # The detector expects a batch of images (numpy array)
        img_for_detection = np.expand_dims(original_face_cv2, axis=0) # Create a batch of 1
        
        try:
            detected_bounding_boxes = detector.get_detections_for_batch(img_for_detection)
        except RuntimeError as e:
             logger.error(f"Face detection runtime error: {e}. If OOM, try smaller image or CPU for detection.")
             raise
        finally:
            del detector # Release detector resources if initialized per session

        if not detected_bounding_boxes or detected_bounding_boxes[0] is None:
            logger.error("Face not detected in the provided image.")
            raise ValueError("Face not detected in the provided image.")

        rect = detected_bounding_boxes[0] # Bounding box for the first (and only) image

        # Calculate crop coordinates with padding
        pady1, pady2, padx1, padx2 = pads
        y1 = max(0, int(rect[1] - pady1))
        y2 = min(original_face_cv2.shape[0], int(rect[3] + pady2))
        x1 = max(0, int(rect[0] - padx1))
        x2 = min(original_face_cv2.shape[1], int(rect[2] + padx2))
        
        face_crop_cv2 = original_face_cv2[y1:y2, x1:x2]
        self.face_coords_in_original_image = (y1, y2, x1, x2) # Store for compositing
        logger.info(f"Face detected and cropped at [{x1},{y1},{x2},{y2}]")

        if face_crop_cv2.size == 0:
            logger.error(f"Face crop is empty. Original shape: {original_face_cv2.shape}, Coords: {(y1,y2,x1,x2)}")
            raise ValueError("Face crop resulted in an empty image.")

        # 3. Prepare the static face tensor for the Wav2Lip model
        face_resized_cv2 = cv2.resize(face_crop_cv2, (IMG_SIZE, IMG_SIZE))
        
        img_masked = face_resized_cv2.copy()
        img_masked[:, IMG_SIZE // 2:] = 0 # Mask bottom half

        # Concatenate masked and original resized face crops (HWC format)
        # Original Wav2Lip preprocesses by stacking along the channel axis (axis=2 for HWC)
        processed_face_hwc = np.concatenate((img_masked, face_resized_cv2), axis=2) / 255.0 # Normalize
        
        # Transpose to CHW for PyTorch and add batch dimension
        self.face_tensor_template = torch.FloatTensor(
            np.transpose(processed_face_hwc, (2, 0, 1)) # HWC to CHW
        ).unsqueeze(0).to(self.device) # Add batch dim: (1, 6, IMG_SIZE, IMG_SIZE)
        logger.info(f"Static face tensor prepared for model: {self.face_tensor_template.shape}")


    def process_audio_chunk(self, audio_chunk_pcm_s16le: bytes) -> list:
        new_audio_s16le = np.frombuffer(audio_chunk_pcm_s16le, dtype=np.int16)
        self.audio_buffer = np.concatenate((self.audio_buffer, new_audio_s16le))
        
        generated_cv2_frames = []

        while len(self.audio_buffer) >= SAMPLES_FOR_MEL_WINDOW:
            # Extract audio segment for one mel window
            current_audio_segment_s16le = self.audio_buffer[:SAMPLES_FOR_MEL_WINDOW]
            
            # Convert S16LE PCM to float32 for melspectrogram function
            # Normalization: Wav2Lip's audio.load_wav normalizes by 32768.0
            current_audio_segment_float32 = current_audio_segment_s16le.astype(np.float32) / 32768.0
            
            # Generate mel spectrogram for this segment using Wav2Lip's audio.py
            try:
                # `audio.melspectrogram` expects a 1D numpy array of float audio samples
                mel_spectrogram = audio.melspectrogram(current_audio_segment_float32) # Shape: (num_mel_filters, num_mel_frames) e.g. (80, T)
            except Exception as e:
                logger.error(f"Error generating mel spectrogram: {e}", exc_info=True)
                break # Stop processing this chunk if mel generation fails

            if mel_spectrogram.shape[1] < MEL_STEP_SIZE:
                logger.warning(f"Mel spectrogram too short ({mel_spectrogram.shape[1]} frames) from {SAMPLES_FOR_MEL_WINDOW} audio samples. Expected at least {MEL_STEP_SIZE}. Waiting for more audio.")
                # This indicates SAMPLES_FOR_MEL_WINDOW might need tuning or audio input is too short.
                # For robust streaming, it might be better to buffer more audio before attempting mel generation
                # or ensure that SAMPLES_FOR_MEL_WINDOW guarantees enough mel frames.
                break 

            # Slice the mel spectrogram to get the desired MEL_STEP_SIZE width
            mel_chunk = mel_spectrogram[:, :MEL_STEP_SIZE] # Shape: (80, MEL_STEP_SIZE)

            # Prepare mel tensor for the model: (Batch, Channels, Mel_filters, Mel_width)
            mel_tensor = torch.FloatTensor(mel_chunk).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 80, MEL_STEP_SIZE)

            # Inference with the pre-processed static face tensor
            with torch.no_grad():
                # Wav2Lip model expects (mel_spectrogram_batch, face_image_batch)
                generated_face_regions_tensor = self.model(mel_tensor, self.face_tensor_template)

            # Post-process the model output
            # Output tensor shape: (Batch, Channels, Height, Width), e.g., (1, 3, IMG_SIZE, IMG_SIZE)
            output_face_region_np = generated_face_regions_tensor.cpu().numpy()[0] # Get first (and only) item from batch
            # Transpose CHW to HWC and denormalize (model output is typically normalized, e.g., -1 to 1 or 0 to 1)
            # Wav2Lip output is usually in range [0,1] if last activation is sigmoid, or needs tanh (-1,1) scaling
            # Assuming output is [0,1] as per original Wav2Lip inference logic * 255.
            output_face_cv2_hwc = (np.transpose(output_face_region_np, (1, 2, 0)) * 255.).astype(np.uint8)

            # Composite the generated face region back onto a copy of the original full-resolution frame
            y1, y2, x1, x2 = self.face_coords_in_original_image
            
            # Resize generated face (IMG_SIZE x IMG_SIZE) to fit the original detected face bounding box
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                 output_face_resized_to_original_box = cv2.resize(output_face_cv2_hwc, (x2 - x1, y2 - y1))
            else:
                logger.warning(f"Invalid face crop dimensions for resize: {(x2-x1), (y2-y1)}. Skipping frame.")
                # Advance audio buffer anyway to prevent getting stuck
                self.audio_buffer = self.audio_buffer[SAMPLES_PER_VIDEO_FRAME:]
                continue


            final_composed_frame = self.original_face_cv2_for_compositing.copy()
            final_composed_frame[y1:y2, x1:x2] = output_face_resized_to_original_box
            generated_cv2_frames.append(final_composed_frame)

            # Advance the audio buffer: remove audio corresponding to ONE video frame's duration
            # This keeps audio consumption paced with video frame generation.
            self.audio_buffer = self.audio_buffer[SAMPLES_PER_VIDEO_FRAME:]
            
        return generated_cv2_frames