import numpy as np
import scipy, cv2, os, sys, audio 
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import platform
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import face_detection 
from models import Wav2Lip
mel_step_size = 16
img_size_global = 96 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} for Wav2Lip inference.')

def _load_wav2lip_checkpoint(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_wav2lip_model(model_path):
    print(f"Loading Wav2Lip model from: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            print("Detected state_dict checkpoint (.pth)")
            model = Wav2Lip()
            s = checkpoint["state_dict"]
            new_s = {k.replace("module.", ""): v for k, v in s.items()}
            model.load_state_dict(new_s)
            return model.to(device).eval()

        elif isinstance(checkpoint, Wav2Lip):
            print("Detected full serialized Wav2Lip model (.pt)")
            return checkpoint.to(device).eval()

    except RuntimeError as e:
        if "PytorchStreamReader failed reading zip archive" not in str(e):
            raise  

    print("Detected TorchScript model (.pt)")
    model = torch.jit.load(model_path, map_location=device)
    return model.eval()

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def detect_faces(images, face_det_batch_size, pads, nosmooth):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                             flip_input=False, device=device)
    
    batch_size = face_det_batch_size
    
    while True:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError as e:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Try using resize_factor.') from e
            batch_size //= 2
            print(f'Recovering from OOM error in face detection; New batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected in one of the frames. Ensure the input image/video contains a clear face.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    
    face_crops_with_coords = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    
    del detector 
    return face_crops_with_coords

def datagen_for_wav2lip(full_frames, mels, box_coords, static_mode, img_size_param, wav2lip_batch_size_param,
                        face_det_batch_size_param, pads_param, nosmooth_param):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box_coords[0] == -1:
        if not static_mode:
            face_crops_with_coords = detect_faces(full_frames, face_det_batch_size_param, pads_param, nosmooth_param)
        else:
            face_crops_with_coords = detect_faces([full_frames[0]], face_det_batch_size_param, pads_param, nosmooth_param)
    else: 
        print('Using the specified bounding box instead of face detection.')
        y1, y2, x1, x2 = box_coords
        face_crops_with_coords = [[frame[y1:y2, x1:x2], (y1, y2, x1, x2)] for frame in full_frames]

    for i, m in enumerate(mels):
        idx = 0 if static_mode else i % len(full_frames)
        original_frame = full_frames[idx].copy()
        face_crop, coords = face_crops_with_coords[idx]

        face_crop_resized = cv2.resize(face_crop, (img_size_param, img_size_param))
            
        img_batch.append(face_crop_resized)
        mel_batch.append(m)
        frame_batch.append(original_frame)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size_param:
            img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch_np.copy()
            img_masked[:, img_size_param//2:] = 0 

            img_batch_processed = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_processed = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])

            yield img_batch_processed, mel_batch_processed, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0: 
        img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch_np.copy()
        img_masked[:, img_size_param//2:] = 0
        img_batch_processed = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
        mel_batch_processed = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
        yield img_batch_processed, mel_batch_processed, frame_batch, coords_batch


def run_wav2lip_inference(
    face_input_path: str,
    audio_input_path: str,
    output_video_path: str,
    checkpoint_path: str,
    wav2lip_model_instance, 
    static: bool = False,
    fps: float = 25.,
    pads: list = [0, 20, 0, 0],
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 128, 
    resize_factor: int = 2,
    crop: list = [0, -1, 0, -1],
    box: list = [-1, -1, -1, -1],
    rotate: bool = False,
    nosmooth: bool = False,
    img_size: int = img_size_global,
    temp_dir: str = "app/temp" 
):
    """
    Core Wav2Lip inference logic.
    Args:
        face_input_path (str): Path to the input image or video file.
        audio_input_path (str): Path to the input audio file (.wav recommended).
        output_video_path (str): Path to save the resulting lip-synced video.
        checkpoint_path (str): Path to the Wav2Lip model checkpoint. (Not used if wav2lip_model_instance is provided)
        wav2lip_model_instance: Pre-loaded Wav2Lip model instance.
        static (bool): If True, use only the first frame of the video/image.
        fps (float): Frames per second for the output video (used if input is static image).
        pads (list): Padding for face detection [top, bottom, left, right].
        face_det_batch_size (int): Batch size for face detection.
        wav2lip_batch_size (int): Batch size for Wav2Lip model.
        resize_factor (int): Factor to resize input video frames.
        crop (list): Crop region for video frames [top, bottom, left, right].
        box (list): Predefined bounding box for the face [top, bottom, left, right]. Use [-1,-1,-1,-1] for auto-detection.
        rotate (bool): If True, rotate video 90 degrees clockwise.
        nosmooth (bool): If True, disable face detection smoothing.
        img_size (int): Size of the face crop fed to Wav2Lip (default 96x96).
        temp_dir (str): Directory for storing temporary files like intermediate audio/video.
    Returns:
        str: Path to the generated output video.
    """

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    
    temp_audio_path = os.path.join(temp_dir, "temp.wav")
    temp_video_path = os.path.join(temp_dir, "result.avi") 

    if os.path.isfile(face_input_path) and face_input_path.lower().split('.')[-1] in ['jpg', 'png', 'jpeg']:
        static_mode_internal = True
    else:
        static_mode_internal = static 

    if not os.path.isfile(face_input_path):
        raise ValueError(f'--face argument must be a valid path to video/image file: {face_input_path}')

    if static_mode_internal:
        full_frames = [cv2.imread(face_input_path)]
        if full_frames[0] is None:
             raise ValueError(f"Failed to read image from {face_input_path}. Check path and image integrity.")
        current_fps = fps 
    else:
        video_stream = cv2.VideoCapture(face_input_path)
        current_fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading video frames...')
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 

            y1_crop, y2_crop, x1_crop, x2_crop = crop
            if x2_crop == -1: x2_crop = frame.shape[1]
            if y2_crop == -1: y2_crop = frame.shape[0]
            frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            full_frames.append(frame)
        if not full_frames:
            raise ValueError(f"Could not read any frames from video: {face_input_path}")

    print(f"Number of frames available for inference: {len(full_frames)}")

    if not audio_input_path.lower().endswith('.wav'):
        print('Input audio is not a .wav file. Converting to .wav...')
        command = f'ffmpeg -y -i "{audio_input_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio_path}"'
        try:
            subprocess.run(command, shell=platform.system() != 'Windows', check=True, capture_output=True, text=True)
            processed_audio_path = temp_audio_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed to convert audio: {e.stderr}") from e
    else:
        processed_audio_path = audio_input_path
    
    wav = audio.load_wav(processed_audio_path, 16000)
    logger.info(f"Shape of wav_data: {wav.shape}, Type: {type(wav)}")
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel spectrogram contains NaN values. If using a TTS voice, try adding a small amount of noise to the .wav file.')

    mel_chunks = []
    mel_idx_multiplier = 80. / current_fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    print(f"Length of mel chunks: {len(mel_chunks)}")

    if len(full_frames) > len(mel_chunks):
        if static_mode_internal: 
            full_frames = [full_frames[0] for _ in range(len(mel_chunks))]
        else:
            full_frames = full_frames[:len(mel_chunks)]
    elif len(mel_chunks) > len(full_frames) and full_frames:
         if static_mode_internal:
            full_frames = [full_frames[0] for _ in range(len(mel_chunks))]
         else:
            last_frame = full_frames[-1]
            full_frames.extend([last_frame for _ in range(len(mel_chunks) - len(full_frames))])


    if not full_frames:
        raise ValueError("No frames available for processing after syncing with audio. Check input video length and audio length.")


    frame_h, frame_w = full_frames[0].shape[:-1]
    video_out_writer = cv2.VideoWriter(temp_video_path,
                                       cv2.VideoWriter_fourcc(*'DIVX'), current_fps, (frame_w, frame_h))

    model = wav2lip_model_instance 

    data_gen = datagen_for_wav2lip(full_frames.copy(), mel_chunks, box, static_mode_internal,
                                   img_size, wav2lip_batch_size, face_det_batch_size, pads, nosmooth)

    for i_batch, (img_batch_processed, mel_batch_processed, original_frames_batch, coords_batch) in enumerate(
                                tqdm(data_gen, total=int(np.ceil(float(len(mel_chunks))/wav2lip_batch_size)))):
        
        img_batch_tensor = torch.FloatTensor(np.transpose(img_batch_processed, (0, 3, 1, 2))).to(device)
        mel_batch_tensor = torch.FloatTensor(np.transpose(mel_batch_processed, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            predicted_faces = model(mel_batch_tensor, img_batch_tensor)
        
        predicted_faces_np = predicted_faces.cpu().numpy().transpose(0, 2, 3, 1) * 255. # Denormalize

        for predicted_face, original_frame, coord_info in zip(predicted_faces_np, original_frames_batch, coords_batch):
            y1, y2, x1, x2 = coord_info
            predicted_face_resized = cv2.resize(predicted_face.astype(np.uint8), (x2 - x1, y2 - y1))
            
            final_frame = original_frame.copy()
            final_frame[y1:y2, x1:x2] = predicted_face_resized
            video_out_writer.write(final_frame)

    video_out_writer.release()

    command = f'ffmpeg -y -i "{temp_video_path}" -i "{processed_audio_path}" -c:v libx264 -c:a aac -strict -2 -q:v 1 "{output_video_path}"'
    try:
        subprocess.run(command, shell=platform.system() != 'Windows', check=True, capture_output=True, text=True)
        print(f"Output video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to combine video and audio: {e.stderr}. Intermediate video at {temp_video_path}") from e

    
    return output_video_path


# if __name__ == '__main__':
    
#     # Create dummy files for testing
#     print("Setting up dummy files for standalone test...")
#     os.makedirs("app/temp", exist_ok=True)
#     os.makedirs("app/checkpoints", exist_ok=True)
#     os.makedirs("results", exist_ok=True)

#     # Dummy checkpoint (replace with your actual checkpoint path)
#     dummy_checkpoint_path = "app/checkpoints/wav2lip_gan.pth"
#     if not os.path.exists(dummy_checkpoint_path):
#         print(f"Warning: Dummy checkpoint {dummy_checkpoint_path} not found. Model loading will fail if not present.")
#         # Create a tiny dummy file to prevent FileNotFoundError if you just want to test script structure
#         # with open(dummy_checkpoint_path, 'w') as f: f.write("dummy checkpoint")


#     # Dummy image (replace with your actual image path)
#     dummy_image_path = "app/temp/sample_face.png"
#     # Create a simple dummy image using OpenCV if none exists
#     if not os.path.exists(dummy_image_path):
#         dummy_img = np.zeros((200, 200, 3), dtype=np.uint8) # Black image
#         cv2.rectangle(dummy_img, (50, 50), (150, 150), (0, 255, 0), -1) # Green square as a "face"
#         cv2.imwrite(dummy_image_path, dummy_img)
#         print(f"Created dummy image at {dummy_image_path}")


#     # Dummy audio (replace with your actual audio path)
#     # For a real test, you need a .wav file.
#     # This creates a very short, silent WAV file for testing purposes.
#     dummy_audio_path = "app/temp/sample_audio.wav"
#     if not os.path.exists(dummy_audio_path):
#         import wave
#         sample_rate = 16000
#         duration = 1 # seconds
#         n_frames = int(sample_rate * duration)
#         comptype = "NONE"
#         compname = "not compressed"
#         n_channels = 1
#         sampwidth = 2 # 16-bit
        
#         with wave.open(dummy_audio_path, 'w') as wf:
#             wf.setparams((n_channels, sampwidth, sample_rate, n_frames, comptype, compname))
#             # Create silent audio data
#             frames = bytearray([0] * n_frames * sampwidth * n_channels)
#             wf.writeframes(frames)
#         print(f"Created dummy audio at {dummy_audio_path}")

#     output_path = "results/test_output.mp4"
    
#     # Test call:
#     try:
#         print("--- Starting Standalone Test ---")
#         # IMPORTANT: For the test to run fully, wav2lip_gan.pth must be present and valid.
#         # The audio.py, face_detection.py, and models.py must also be correctly in the PYTHONPATH.
#         if os.path.exists(dummy_checkpoint_path): # Only run if model might load
#             # Load the model once before calling the inference function
#             loaded_model = load_wav2lip_model(dummy_checkpoint_path)
            
#             run_wav2lip_inference(
#                 face_input_path=dummy_image_path,
#                 audio_input_path=dummy_audio_path,
#                 output_video_path=output_path,
#                 checkpoint_path=dummy_checkpoint_path, # Still needed for consistency, though model is passed
#                 wav2lip_model_instance=loaded_model,
#                 static=True, # Since input is an image
#                 # fps=25., pads=[0,10,0,0] ... other params can be added
#                 pads = [0, 10, 0, 0],
#                 resize_factor=2,

#             )
#             print(f"Standalone test completed. Output should be at {output_path}")
#         else:
#             print(f"Skipping full test execution as checkpoint {dummy_checkpoint_path} is missing.")

#     except Exception as e:
#         print(f"Error during standalone test: {e}")
#         import traceback
#         traceback.print_exc()