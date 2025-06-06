import mimetypes
import os
import tempfile
from .wav2lip_inference_logic import run_wav2lip_inference, load_wav2lip_model

WAV2LIP_MODEL = None
WAV2LIP_CHECKPOINT_PATH = os.environ.get("WAV2LIP_CHECKPOINT", "app/checkpoints/wav2lip_gan.pth")

def load_model_on_startup(): 
    global WAV2LIP_MODEL
    if not os.path.exists(WAV2LIP_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Wav2Lip checkpoint not found at {WAV2LIP_CHECKPOINT_PATH}. Please ensure it's correctly placed.")
    WAV2LIP_MODEL = load_wav2lip_model(WAV2LIP_CHECKPOINT_PATH)
    print("Wav2Lip model loaded successfully.")

async def generate_lip_sync_video(face_bytes: bytes, audio_bytes: bytes, face_type: str) -> bytes:
    global WAV2LIP_MODEL
    if WAV2LIP_MODEL is None:
        print("Warning: Wav2Lip model not pre-loaded. Loading now (this will be slow per request)...")
        load_model_on_startup()
        if WAV2LIP_MODEL is None:
             raise RuntimeError("Wav2Lip model could not be loaded.")

    is_static = face_type.startswith('image/')
    
    face_suffix = mimetypes.guess_extension(face_type) or '.tmp'

    with tempfile.NamedTemporaryFile(suffix=face_suffix, delete=False) as tmp_face_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output_video_file:
        
        tmp_face_path = tmp_face_file.name
        tmp_face_file.write(face_bytes)
        
        tmp_audio_path = tmp_audio_file.name
        tmp_audio_file.write(audio_bytes)

        output_video_actual_path = tmp_output_video_file.name
        
    output_video_bytes = None
    try:
        print(f"Starting Wav2Lip processing: Input at {tmp_face_path}, Audio at {tmp_audio_path}")
        
        run_wav2lip_inference(
            face_input_path=tmp_face_path,
            audio_input_path=tmp_audio_path,
            output_video_path=output_video_actual_path,
            checkpoint_path=WAV2LIP_CHECKPOINT_PATH,
            wav2lip_model_instance=WAV2LIP_MODEL,
            static=is_static,
            temp_dir=os.path.join(os.getcwd(), "app", "temp")
        )

        print(f"Wav2Lip processing finished. Output video at: {output_video_actual_path}")
        with open(output_video_actual_path, "rb") as f_video:
            output_video_bytes = f_video.read()
        
    except Exception as e:
        print(f"Error during Wav2Lip processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Wav2Lip processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_face_path):
            os.remove(tmp_face_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        if os.path.exists(output_video_actual_path):
            os.remove(output_video_actual_path) 
    if not output_video_bytes:
        raise RuntimeError("Video generation failed or produced an empty file.")
        
    return output_video_bytes