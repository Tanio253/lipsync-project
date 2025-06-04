from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import base64
import json
import logging
import os

# Assuming wav2lip_handler is in the same 'app' package
from .wav2lip_handler import generate_lip_sync_video, load_model_on_startup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="LipSync WebSocket API")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    try:
        # Set an environment variable to indicate running in Docker if this helps logic elsewhere
        # This is just an example, might not be strictly necessary if paths are handled well.
        os.environ['DOCKER_CONTAINER'] = '1'
        load_model_on_startup() # [cite: 8]
    except Exception as e:
        logger.error(f"FATAL: Could not load Wav2Lip model on startup: {e}", exc_info=True)
        # Depending on desired behavior, you might want the app to fail starting
        # or handle this gracefully (e.g., API returns error until model loads).
        # For now, it will raise the exception and FastAPI might not start fully.

# Simple HTML page for testing WebSocket
html_test_page = """
<!DOCTYPE html>
<html>
    <head>
        <title>LipSync WebSocket Test</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
            h1 { text-align: center; color: #333; }
            .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .controls, .video-area { margin-bottom: 20px; }
            label, button, input { margin-top: 10px; margin-bottom: 10px; padding: 8px; border-radius: 4px; border: 1px solid #ddd; }
            button { background-color: #007bff; color: white; cursor: pointer; border-color: #007bff; }
            button:hover { background-color: #0056b3; }
            #status { margin-top: 15px; padding: 10px; background-color: #e9ecef; border-radius: 4px; }
            video { display: block; margin-top: 10px; max-width: 100%; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WebSocket LipSync Test</h1>
            <div class="controls">
                <label for="imageFile">Choose Image:</label>
                <input type="file" id="imageFile" accept="image/png, image/jpeg">
                <br>
                <label for="audioFile">Choose Audio:</label>
                <input type="file" id="audioFile" accept="audio/wav, audio/mp3">
                <br>
                <button onclick="sendMessage()">Generate LipSync Video</button>
            </div>
            <div id="status">Awaiting connection...</div>
            <div class="video-area">
                <video id="outputVideo" controls autoplay playsinline></video>
            </div>
        </div>
        <script>
            var ws;
            const statusDiv = document.getElementById('status');
            const outputVideo = document.getElementById('outputVideo');
            const imageFileInput = document.getElementById('imageFile');
            const audioFileInput = document.getElementById('audioFile');

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.host;
                ws = new WebSocket(`${protocol}//${host}/ws/lipsync`);

                ws.onopen = function(event) {
                    statusDiv.innerHTML = "WebSocket Connected. Ready to send data.";
                    statusDiv.style.backgroundColor = '#d4edda'; // Greenish for success
                };

                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.error) {
                            statusDiv.innerHTML = "Error: " + data.error;
                            statusDiv.style.backgroundColor = '#f8d7da'; // Reddish for error
                        } else if (data.video_base64) { // [cite: 3]
                            outputVideo.src = "data:video/mp4;base64," + data.video_base64;
                            outputVideo.load(); // Important to load new src
                            outputVideo.play().catch(e => console.warn("Autoplay was prevented:", e));
                            statusDiv.innerHTML = "Video received and displayed!";
                            statusDiv.style.backgroundColor = '#d4edda';
                        } else if (data.message) {
                             statusDiv.innerHTML = "Server: " + data.message;
                             statusDiv.style.backgroundColor = '#e9ecef';
                        }
                    } catch (e) {
                        statusDiv.innerHTML = "Received non-JSON message or parse error: " + event.data;
                        statusDiv.style.backgroundColor = '#fff3cd'; // Yellowish for warning
                        console.error("WebSocket message error:", e, "Raw data:", event.data);
                    }
                };

                ws.onclose = function(event) {
                    statusDiv.innerHTML = "WebSocket Disconnected. Attempting to reconnect in 3 seconds...";
                    statusDiv.style.backgroundColor = '#fff3cd';
                    setTimeout(connect, 3000); // Try to reconnect
                };

                ws.onerror = function(event) {
                    statusDiv.innerHTML = "WebSocket Error. Check console.";
                    statusDiv.style.backgroundColor = '#f8d7da';
                    console.error("WebSocket Error: ", event);
                };
            }

            function getBase64(file, callback) {
                var reader = new FileReader();
                reader.readAsDataURL(file); // This reads the file as a data URL (includes mime type)
                reader.onload = function () {
                    // The result includes "data:mime/type;base64,", we only want the part after comma
                    const base64String = reader.result.split(',')[1];
                    callback(base64String, file.type);
                };
                reader.onerror = function (error) {
                    console.log('Error reading file: ', error);
                    statusDiv.innerHTML = "Error reading file: " + error;
                    statusDiv.style.backgroundColor = '#f8d7da';
                };
            }

            function sendMessage() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    statusDiv.innerHTML = "WebSocket is not connected. Please wait or try refreshing.";
                    statusDiv.style.backgroundColor = '#f8d7da';
                    return;
                }

                const imageFile = imageFileInput.files[0];
                const audioFile = audioFileInput.files[0];

                if (!imageFile || !audioFile) {
                    statusDiv.innerHTML = "Please select both an image and an audio file.";
                    statusDiv.style.backgroundColor = '#fff3cd';
                    return;
                }

                statusDiv.innerHTML = "Reading files and preparing to send data...";
                statusDiv.style.backgroundColor = '#e9ecef';

                getBase64(imageFile, function(imageBase64, imageType) {
                    getBase64(audioFile, function(audioBase64, audioType) {
                        const payload = {
                            image_base64: imageBase64, // [cite: 2]
                            audio_base64: audioBase64, // [cite: 2]
                            image_type: imageType, // e.g. "image/png"
                            audio_type: audioType  // e.g. "audio/wav"
                        };
                        ws.send(JSON.stringify(payload));
                        statusDiv.innerHTML = "Data sent to server. Waiting for processing (this can take some time)...";
                        statusDiv.style.backgroundColor = '#cfe2ff'; // Bluish for processing
                    });
                });
            }
            
            // Initial connection attempt
            connect();
        </script>
    </body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_test_client_page():
    return html_test_page

@app.websocket("/ws/lipsync") # [cite: 2, 5]
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted from client.")
    try:
        while True:
            data_str = await websocket.receive_text()
            logger.debug(f"Received raw data string of length: {len(data_str)}")
            
            try:
                data = json.loads(data_str)
                image_b64 = data.get("image_base64")
                audio_b64 = data.get("audio_base64")
                # Optional: get file extensions if needed by handler
                image_type = data.get("image_type", "image/png") # Default to png
                audio_type = data.get("audio_type", "audio/wav") # Default to wav

                # Simple validation for file extensions based on mime types
                image_ext = "." + image_type.split('/')[-1] if image_type else ".png"
                audio_ext = "." + audio_type.split('/')[-1] if audio_type else ".wav"
                # More robust extension mapping might be needed
                if image_ext == ".jpeg": image_ext = ".jpg"


                if not image_b64 or not audio_b64:
                    logger.warning("Missing image_base64 or audio_base64 in request.")
                    await websocket.send_json({"error": "Missing image_base64 or audio_base64"})
                    continue

                logger.info("Decoding base64 image and audio.")
                image_bytes = base64.b64decode(image_b64)
                audio_bytes = base64.b64decode(audio_b64)
                
                await websocket.send_json({"message": "Data received. Starting lip sync generation..."})
                logger.info("Starting lip sync video generation...")
                
                video_bytes = await generate_lip_sync_video(image_bytes, audio_bytes) # [cite: 3, 4]
                
                logger.info("Lip sync video generation complete. Encoding to base64 and sending.")
                video_b64 = base64.b64encode(video_bytes).decode('utf-8')
                
                await websocket.send_json({"video_base64": video_b64}) # [cite: 3]
                logger.info("Video sent to client.")

            except json.JSONDecodeError:
                logger.error("Invalid JSON received.")
                await websocket.send_json({"error": "Invalid JSON format received."})
            except base64.binascii.Error as b64_error:
                logger.error(f"Base64 decoding error: {b64_error}")
                await websocket.send_json({"error": f"Invalid base64 data: {b64_error}"})
            except FileNotFoundError as fnf_error: # Catch specific errors from handler
                logger.error(f"File not found during processing: {fnf_error}", exc_info=True)
                await websocket.send_json({"error": f"Server configuration error: {fnf_error}"})
            except RuntimeError as processing_error: # Catch specific errors from handler
                logger.error(f"Processing error: {processing_error}", exc_info=True)
                await websocket.send_json({"error": f"Video generation failed: {processing_error}"})
            except Exception as e:
                logger.error(f"Unhandled error processing WebSocket message: {str(e)}", exc_info=True)
                await websocket.send_json({"error": f"An unexpected server error occurred: {str(e)}"})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client.")
    except Exception as e:
        logger.error(f"Unhandled WebSocket exception: {str(e)}", exc_info=True)
        # Try to close gracefully if possible
        try:
            await websocket.close(code=1011, reason=f"Unhandled server error: {str(e)}")
        except Exception:
            pass # Websocket might already be closed or in a bad state
    finally:
        logger.info("WebSocket endpoint handler finished.")

# To run this app (from the parent directory of 'app', i.e., 'lipsync_project'):
# uvicorn app.main:app --reload --port 8000