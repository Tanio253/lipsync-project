# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import base64
import json
import logging
import os
import numpy as np
import cv2 # For encoding frames to JPEG
import torch
# Assuming your Wav2Lip model loading logic is in wav2lip_handler or similar
# from .wav2lip_handler import load_model_on_startup # Your existing model loader
# For this example, let's define how WAV2LIP_MODEL would be loaded conceptually
# You need to ensure 'models.py' (with Wav2Lip class) and 'face_detection.py' are in path
from models import Wav2Lip # From Wav2Lip repository
from .wav2lip_inference_logic import load_wav2lip_model # User's existing loader

from .streaming_processor import Wav2LipStreamingSession # The new class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-Time LipSync WebSocket API")

# --- Global Wav2Lip Model ---
WAV2LIP_MODEL = None
# Define path to your Wav2Lip checkpoint (e.g., wav2lip_gan.pth)
# This should be an environment variable or a config value
WAV2LIP_CHECKPOINT_PATH = os.environ.get("WAV2LIP_CHECKPOINT", "app/checkpoints/wav2lip_gan.pth")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # From wav2lip_inference_logic.py

@app.on_event("startup")
async def startup_event():
    global WAV2LIP_MODEL
    logger.info("Application startup...")
    logger.info(f"Using device: {DEVICE}")
    try:
        if not os.path.exists(WAV2LIP_CHECKPOINT_PATH):
            logger.error(f"FATAL: Wav2Lip checkpoint not found at {WAV2LIP_CHECKPOINT_PATH}")
            # Application might still start, but streaming sessions will fail.
            # Consider raising an error to prevent FastAPI from starting if model is critical.
            raise FileNotFoundError(f"Wav2Lip checkpoint not found: {WAV2LIP_CHECKPOINT_PATH}")
        
        # Using the loading logic from user's wav2lip_inference_logic.py
        WAV2LIP_MODEL = load_wav2lip_model(WAV2LIP_CHECKPOINT_PATH)
        # Ensure model is in eval mode and on the correct device (load_wav2lip_model should handle .to(device).eval())
        WAV2LIP_MODEL = WAV2LIP_MODEL.to(DEVICE).eval()

        logger.info("Wav2Lip model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load Wav2Lip model on startup: {e}", exc_info=True)
        # This will likely prevent the app from working correctly.
        # raise # Optionally re-raise to stop FastAPI startup

# Store active streaming sessions
active_sessions = {} # Key: client_id (e.g., "host:port"), Value: Wav2LipStreamingSession instance

# --- HTML Test Page ---
# (This will be more complex to handle raw PCM audio capture and streaming)
html_test_page_streaming = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time LipSync Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; display: flex; flex-direction: column; align-items: center; }
        .container { background-color: #fff; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); width: 90%; max-width: 700px; }
        h1 { color: #333; text-align: center; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; color: #555; }
        input[type="file"], button { padding: 10px; margin-top: 5px; border-radius: 5px; border: 1px solid #ddd; width: calc(100% - 22px); box-sizing: border-box; }
        button { background-color: #007bff; color: white; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #status { margin-top: 20px; padding: 12px; background-color: #e9ecef; border-radius: 5px; text-align: center; font-weight: bold; }
        #videoArea { margin-top: 20px; text-align: center; }
        #outputImage { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 5px; background-color: #eee; min-height: 200px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Real-Time LipSync Streaming</h1>
        
        <label for="imageFile">1. Choose Face Image (PNG/JPEG):</label>
        <input type="file" id="imageFile" accept="image/png, image/jpeg">

        <button id="startSessionButton" style="margin-top: 20px;">2. Start LipSync Session</button>
        
        <label for="audioControls" style="margin-top: 20px;">3. Audio Streaming:</label>
        <div id="audioControls">
            <button id="startRecordButton" disabled>Start Recording & Streaming</button>
            <button id="stopRecordButton" disabled>Stop Recording</button>
        </div>
        
        <div id="status">Please select an image and start session.</div>
        
        <div id="videoArea">
            <img id="outputImage" alt="Lip-synced video stream will appear here">
        </div>
    </div>

    <script>
        const imageFileInput = document.getElementById('imageFile');
        const startSessionButton = document.getElementById('startSessionButton');
        const startRecordButton = document.getElementById('startRecordButton');
        const stopRecordButton = document.getElementById('stopRecordButton');
        const statusDiv = document.getElementById('status');
        const outputImage = document.getElementById('outputImage');

        let ws;
        let audioContext;
        let scriptProcessor; // Or AudioWorkletNode
        let mediaStreamSource;
        let imageBase64 = null;
        const TARGET_SAMPLE_RATE = 16000;
        const BUFFER_SIZE = 16384; // Audio buffer size for ScriptProcessorNode

        imageFileInput.onchange = function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imageBase64 = e.target.result.split(',')[1]; // Get base64 part
                    statusDiv.textContent = "Image selected. Ready to start session.";
                }
                reader.readAsDataURL(file);
            }
        };

        startSessionButton.onclick = function() {
            if (!imageBase64) {
                alert("Please select an image first.");
                return;
            }
            if (ws && ws.readyState === WebSocket.OPEN) {
                statusDiv.textContent = "Session already active. Stop recording to start a new one.";
                return;
            }
            connectWebSocket();
        };
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            ws = new WebSocket(`${protocol}//${host}/ws/lipsync_stream`);

            ws.onopen = function() {
                statusDiv.textContent = "WebSocket connected. Sending image to start session...";
                ws.send(JSON.stringify({ type: "start_session", image_base64: imageBase64 }));
                startSessionButton.disabled = true;
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.error) {
                    statusDiv.textContent = "Error: " + data.error;
                    console.error("WebSocket Error:", data.error);
                    stopRecording(); // Stop recording on error
                    startRecordButton.disabled = true;
                    startSessionButton.disabled = false; // Allow restarting session
                } else if (data.status && data.status.includes("Session started")) {
                    statusDiv.textContent = "Session started! Ready to stream audio.";
                    startRecordButton.disabled = false;
                    stopRecordButton.disabled = true;
                } else if (data.type === "video_frame" && data.frame_base64) {
                    outputImage.src = "data:image/jpeg;base64," + data.frame_base64;
                }
            };

            ws.onclose = function() {
                statusDiv.textContent = "WebSocket disconnected.";
                console.log("WebSocket disconnected");
                stopRecording(); // Ensure recording stops
                startRecordButton.disabled = true;
                stopRecordButton.disabled = true;
                startSessionButton.disabled = false;
            };

            ws.onerror = function(error) {
                statusDiv.textContent = "WebSocket error. See console.";
                console.error("WebSocket Error: ", error);
                stopRecording();
            };
        }

        startRecordButton.onclick = async function() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                alert("WebSocket not connected. Please start session.");
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SAMPLE_RATE });
                
                // Check if desired sample rate was achieved
                if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
                    console.warn(`Requested ${TARGET_SAMPLE_RATE}Hz but got ${audioContext.sampleRate}Hz. Resampling might be needed or quality affected.`);
                    // For simplicity, this example proceeds. Robust solution might resample or inform user.
                }

                mediaStreamSource = audioContext.createMediaStreamSource(stream);
                scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1); // 1 input channel, 1 output channel

                scriptProcessor.onaudioprocess = function(audioProcessingEvent) {
                    if (!ws || ws.readyState !== WebSocket.OPEN) return;

                    const inputBuffer = audioProcessingEvent.inputBuffer;
                    const pcmFloat32Data = inputBuffer.getChannelData(0); // Mono

                    // Convert Float32 PCM to Int16 PCM
                    const pcmInt16Data = new Int16Array(pcmFloat32Data.length);
                    for (let i = 0; i < pcmFloat32Data.length; i++) {
                        let val = pcmFloat32Data[i] * 32767; // Scale to 16-bit range
                        val = Math.max(-32768, Math.min(32767, val)); // Clamp
                        pcmInt16Data[i] = val;
                    }
                    
                    // Convert Int16Array to base64 string
                    // Need to access the underlying ArrayBuffer and then convert to base64
                    const audioChunkBase64 = btoa(String.fromCharCode.apply(null, new Uint8Array(pcmInt16Data.buffer)));
                    
                    if (ws.readyState === WebSocket.OPEN) {
                         ws.send(JSON.stringify({ type: "audio_chunk", audio_chunk_base64: audioChunkBase64 }));
                    }
                };

                mediaStreamSource.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination); // Connect to output, though not strictly needed if just capturing

                statusDiv.textContent = "Recording and streaming audio...";
                startRecordButton.disabled = true;
                stopRecordButton.disabled = false;
                imageFileInput.disabled = true; // Prevent changing image during session

            } catch (err) {
                console.error("Error starting audio recording:", err);
                statusDiv.textContent = "Error starting audio: " + err.message;
                alert("Could not start audio recording: " + err.message + " Ensure microphone access is allowed.");
            }
        };

        stopRecordButton.onclick = function() {
            stopRecording();
            statusDiv.textContent = "Recording stopped. Session still open for new recording or close WebSocket.";
            startRecordButton.disabled = false; // Allow restarting recording
            imageFileInput.disabled = false;
            if (ws && ws.readyState === WebSocket.OPEN) {
                // Optionally send a "stop_audio_stream" message to server if needed
            }
        };
        
        function stopRecording() {
            if (scriptProcessor) {
                scriptProcessor.disconnect();
                scriptProcessor = null;
            }
            if (mediaStreamSource) {
                mediaStreamSource.disconnect();
                // Stop all tracks on the stream to release microphone
                if (mediaStreamSource.mediaStream && mediaStreamSource.mediaStream.getTracks) {
                    mediaStreamSource.mediaStream.getTracks().forEach(track => track.stop());
                }
                mediaStreamSource = null;
            }
            if (audioContext && audioContext.state !== 'closed') {
                audioContext.close();
                audioContext = null;
            }
            stopRecordButton.disabled = true;
            // Do not disable startRecordButton if WebSocket is still open
            if (ws && ws.readyState === WebSocket.OPEN) {
                startRecordButton.disabled = false;
            } else {
                 startRecordButton.disabled = true; // No WebSocket, no recording
            }
        }

    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_streaming_test_client_page():
    return HTMLResponse(html_test_page_streaming)

@app.websocket("/ws/lipsync_stream")
async def websocket_streaming_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Streaming WebSocket connection accepted from {client_id}")
    session: Wav2LipStreamingSession = None

    try:
        while True:
            # Using receive_text and json.loads for more flexibility if non-JSON messages are possible for control
            raw_data = await websocket.receive_text()
            try:
                message = json.loads(raw_data)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from {client_id}: {raw_data}")
                await websocket.send_json({"error": "Invalid JSON format."})
                continue

            message_type = message.get("type")

            if message_type == "start_session":
                image_b64 = message.get("image_base64")
                if not image_b64:
                    await websocket.send_json({"error": "No image_base64 in start_session."})
                    continue
                
                if WAV2LIP_MODEL is None:
                    logger.error(f"Attempt to start session for {client_id} but model is not loaded.")
                    await websocket.send_json({"error": "Server model not ready. Please try again later."})
                    break # Critical server error, close connection

                try:
                    image_bytes = base64.b64decode(image_b64)
                    session = Wav2LipStreamingSession(WAV2LIP_MODEL, image_bytes, device=DEVICE)
                    active_sessions[client_id] = session
                    await websocket.send_json({"status": "Session started, ready for audio."})
                    logger.info(f"Session started for {client_id}")
                except Exception as e:
                    logger.error(f"Error starting session for {client_id}: {e}", exc_info=True)
                    await websocket.send_json({"error": f"Session start error: {str(e)}"})
                    break 

            elif message_type == "audio_chunk":
                if not session:
                    await websocket.send_json({"error": "Session not started. Send start_session first."})
                    continue
                
                audio_chunk_b64 = message.get("audio_chunk_base64")
                if not audio_chunk_b64:
                    await websocket.send_json({"error": "No audio_chunk_base64 in audio_chunk."})
                    continue
                
                try:
                    # Client should send raw Int16 PCM bytes, base64 encoded.
                    audio_chunk_pcm_s16le = base64.b64decode(audio_chunk_b64)
                    
                    video_frames_cv2 = session.process_audio_chunk(audio_chunk_pcm_s16le)
                    
                    if not video_frames_cv2: # No new frames generated yet (e.g. buffer filling)
                        continue

                    for frame_cv2 in video_frames_cv2:
                        # Encode frame to JPEG for streaming
                        # Quality can be adjusted (0-100, default 95 for JPEG)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        _, buffer = cv2.imencode('.jpg', frame_cv2, encode_param)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        await websocket.send_json({"type": "video_frame", "frame_base64": frame_b64})
                
                except base64.binascii.Error as b64_err:
                    logger.warning(f"Invalid base64 audio data from {client_id}: {b64_err}")
                    await websocket.send_json({"error": f"Invalid base64 audio data: {b64_err}"})
                except Exception as e:
                    logger.error(f"Error processing audio chunk for {client_id}: {e}", exc_info=True)
                    await websocket.send_json({"error": f"Audio processing error: {str(e)}"}) # Send general error

            else:
                logger.warning(f"Unknown message type received from {client_id}: {message_type}")
                await websocket.send_json({"error": f"Unknown message type: {message_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket {client_id} disconnected.")
    except Exception as e: # Catch any other unexpected errors in the loop
        logger.error(f"Unhandled WebSocket error for {client_id}: {e}", exc_info=True)
        # Try to send a final error message if the websocket is still somewhat open
        try:
            await websocket.send_json({"error": f"Unhandled server error: {str(e)}"})
        except Exception:
            pass # Ignore if sending fails
    finally:
        if client_id in active_sessions:
            del active_sessions[client_id]
            logger.info(f"Cleaned up session for {client_id}")
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"Streaming WebSocket handler for {client_id} finished.")

# To run this (from parent directory of 'app'):
# uvicorn app.main:app --reload --port 8000
# Ensure WAV2LIP_CHECKPOINT environment variable is set or path is correct.
# Ensure Wav2Lip's audio.py, face_detection.py, models.py are in the PYTHONPATH
# (e.g., by placing them in the 'app' directory or installing Wav2Lip as a package).