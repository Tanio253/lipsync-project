import cv2
import numpy as np
import torch
import logging

import audio
import face_detection

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_FFT = 800
HOP_LENGTH = 200
FPS = 25
MEL_STEP_SIZE = 16
IMG_SIZE = 96
SAMPLES_PER_VIDEO_FRAME = int(SAMPLE_RATE / FPS)
SAMPLES_FOR_MEL_WINDOW = (MEL_STEP_SIZE - 1) * HOP_LENGTH + N_FFT
MAX_BUFFER_DURATION_S = 0.5
MAX_BUFFER_SAMPLES = int(SAMPLE_RATE * MAX_BUFFER_DURATION_S)


class Wav2LipStreamingSession:
    """A state container for a single client's lip-sync session."""
    def __init__(self, face_tensor_template: torch.Tensor, face_coords: tuple, original_face_cv2: np.ndarray):
        """
        Initializes the session with pre-processed face data.
        The model itself is managed by the worker, not the session.
        """
        self.audio_buffer = np.array([], dtype=np.int16)
        self.face_tensor_template = face_tensor_template
        self.face_coords_in_original_image = face_coords
        self.original_face_cv2_for_compositing = original_face_cv2
        self.target_resize_dims = (face_coords[3] - face_coords[2], face_coords[1] - face_coords[0])

    def add_audio_chunk(self, audio_chunk_pcm_s16le: bytes):
        """
        Appends a new audio chunk to the internal buffer.
        """
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            logger.warning(f"Audio buffer is too long ({len(self.audio_buffer)} samples > {MAX_BUFFER_SAMPLES}). "
                           "Clearing buffer to catch up to real-time.")
            self.audio_buffer = np.array([], dtype=np.int16)

    
        new_audio_s16le = np.frombuffer(audio_chunk_pcm_s16le, dtype=np.int16)
        self.audio_buffer = np.concatenate((self.audio_buffer, new_audio_s16le))

    def get_available_mel_chunks(self) -> list:
        """Processes the buffer and returns generated mel spectrogram chunks."""
        mel_chunks = []
        while len(self.audio_buffer) >= SAMPLES_FOR_MEL_WINDOW:
            audio_segment_s16le = self.audio_buffer[:SAMPLES_FOR_MEL_WINDOW]
            audio_segment_float32 = audio_segment_s16le.astype(np.float32) / 32768.0
            
            mel_spectrogram = audio.melspectrogram(audio_segment_float32)

            if mel_spectrogram.shape[1] >= MEL_STEP_SIZE:
                mel_chunks.append(mel_spectrogram[:, :MEL_STEP_SIZE])

            self.audio_buffer = self.audio_buffer[SAMPLES_PER_VIDEO_FRAME:]
        
        return mel_chunks

    def composite_frame(self, generated_face_region_np: np.ndarray) -> np.ndarray:
        """Composites the generated face onto the original image."""
        output_face_cv2_hwc = (np.transpose(generated_face_region_np, (1, 2, 0)) * 255.).astype(np.uint8)
        output_face_resized = cv2.resize(output_face_cv2_hwc, self.target_resize_dims)
        y1, y2, x1, x2 = self.face_coords_in_original_image
        final_composed_frame = self.original_face_cv2_for_compositing.copy()
        final_composed_frame[y1:y2, x1:x2] = output_face_resized
        return final_composed_frame