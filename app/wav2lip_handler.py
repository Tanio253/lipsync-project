# In app/wav2lip_handler.py

# ... other imports ...
from .wav2lip_inference_logic import run_wav2lip_inference, load_wav2lip_model # Assuming it's in the same 'app' directory
import tempfile
import os

# --- Global variable for the loaded model ---
# This model should be loaded once when your FastAPI app starts.
# You can use FastAPI's startup event for this.
WAV2LIP_MODEL = None
WAV2LIP_CHECKPOINT_PATH = os.environ.get("WAV2LIP_CHECKPOINT", "app/checkpoints/wav2lip_gan.pth")

def load_model_on_startup(): # Call this function during FastAPI startup
    global WAV2LIP_MODEL
    if not os.path.exists(WAV2LIP_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Wav2Lip checkpoint not found at {WAV2LIP_CHECKPOINT_PATH}. Please ensure it's correctly placed.")
    WAV2LIP_MODEL = load_wav2lip_model(WAV2LIP_CHECKPOINT_PATH)
    print("Wav2Lip model loaded successfully.")

# In your FastAPI main.py, you'd do something like:
# app = FastAPI()
# @app.on_event("startup")
# async def startup_event():
#     from app.wav2lip_handler import load_model_on_startup
#     load_model_on_startup()


async def generate_lip_sync_video(image_bytes: bytes, audio_bytes: bytes) -> bytes:
    global WAV2LIP_MODEL
    if WAV2LIP_MODEL is None:
        # Fallback if startup loading failed or wasn't implemented, though not ideal for performance
        print("Warning: Wav2Lip model not pre-loaded. Loading now (this will be slow per request)...")
        load_model_on_startup() # Or handle error appropriately
        if WAV2LIP_MODEL is None: # If still None after attempting load
             raise RuntimeError("Wav2Lip model could not be loaded.")


    # Create temporary files for image, audio, and output video
    # Suffixes are important for Wav2Lip and ffmpeg to correctly identify file types
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file, \
         tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output_video_file:
        
        tmp_image_path = tmp_image_file.name
        tmp_image_file.write(image_bytes)
        
        tmp_audio_path = tmp_audio_file.name
        tmp_audio_file.write(audio_bytes)

        # The output path from run_wav2lip_inference will be where the final .mp4 is stored
        # For simplicity, we can use the tmp_output_video_file.name directly if run_wav2lip_inference
        # writes to the path we provide.
        output_video_actual_path = tmp_output_video_file.name
        # We close it here because run_wav2lip_inference will open/write to it.
        # On Windows, files often cannot be opened by multiple processes/handles simultaneously.
        
    output_video_bytes = None
    try:
        print(f"Starting Wav2Lip processing: Image at {tmp_image_path}, Audio at {tmp_audio_path}")
        
        # Key change: Pass the pre-loaded WAV2LIP_MODEL
        run_wav2lip_inference(
            face_input_path=tmp_image_path,
            audio_input_path=tmp_audio_path, # Ensure this audio is in a format ffmpeg can read, or .wav
            output_video_path=output_video_actual_path,
            checkpoint_path=WAV2LIP_CHECKPOINT_PATH, # Still passed for consistency, though model is pre-loaded
            wav2lip_model_instance=WAV2LIP_MODEL,
            static=True, # Assuming single image input, adjust if video input is allowed for 'face'
            # You can expose other parameters like pads, resize_factor etc., through your API if needed
            temp_dir=os.path.join(os.getcwd(), "app", "temp") # Ensure this directory exists and is writable
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
        # Clean up temporary files
        if os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        if os.path.exists(output_video_actual_path): # This is the final output, might be kept if debugging
            os.remove(output_video_actual_path) 
            # If you want to serve the file via URL instead of base64, don't delete it here.

    if not output_video_bytes:
        raise RuntimeError("Video generation failed or produced an empty file.")
        
    return output_video_bytes