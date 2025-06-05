from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import base64
import json
import logging
import os

from .wav2lip_handler import generate_lip_sync_video, load_model_on_startup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="LipSync WebSocket API")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    try:
        os.environ['DOCKER_CONTAINER'] = '1'
        load_model_on_startup()
    except Exception as e:
        logger.error(f"FATAL: Could not load Wav2Lip model on startup: {e}", exc_info=True)

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
            <h1>WebSocket LipSync Test (Image or Video)</h1>
            <div class="controls">
                <label for="faceFile">Choose Image or Video:</label>
                <input type="file" id="faceFile" accept="image/png, image/jpeg, video/mp4, video/quicktime">
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
            const faceFileInput = document.getElementById('faceFile');
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
                        } else if (data.video_base64) {
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

                const faceFile = faceFileInput.files[0];
                const audioFile = audioFileInput.files[0];

                if (!faceFile || !audioFile) {
                    statusDiv.innerHTML = "Please select both a face input (image/video) and an audio file.";
                    statusDiv.style.backgroundColor = '#fff3cd';
                    return;
                }

                statusDiv.innerHTML = "Reading files and preparing to send data...";
                statusDiv.style.backgroundColor = '#e9ecef';

                getBase64(faceFile, function(faceBase64, faceType) {
                    getBase64(audioFile, function(audioBase64, audioType) {
                        const payload = {
                            face_base64: faceBase64,
                            audio_base64: audioBase64,
                            face_type: faceType, // e.g. "image/png" or "video/mp4"
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

@app.websocket("/ws/lipsync")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted from client.")
    try:
        while True:
            data_str = await websocket.receive_text()
            logger.debug(f"Received raw data string of length: {len(data_str)}")
            
            try:
                data = json.loads(data_str)
                face_b64 = data.get("face_base64")
                audio_b64 = data.get("audio_base64")
                face_type = data.get("face_type", "image/png") 
                audio_type = data.get("audio_type", "audio/wav") 

                if not face_b64 or not audio_b64:
                    logger.warning("Missing face_base64 or audio_base64 in request.")
                    await websocket.send_json({"error": "Missing face_base64 or audio_base64"})
                    continue

                logger.info(f"Decoding base64 face input ({face_type}) and audio.")
                face_bytes = base64.b64decode(face_b64)
                audio_bytes = base64.b64decode(audio_b64)
                
                await websocket.send_json({"message": "Data received. Starting lip sync generation..."})
                logger.info("Starting lip sync video generation...")
                
                video_bytes = await generate_lip_sync_video(face_bytes, audio_bytes, face_type)
                
                logger.info("Lip sync video generation complete. Encoding to base64 and sending.")
                video_b64 = base64.b64encode(video_bytes).decode('utf-8')
                
                await websocket.send_json({"video_base64": video_b64})
                logger.info("Video sent to client.")

            except json.JSONDecodeError:
                logger.error("Invalid JSON received.")
                await websocket.send_json({"error": "Invalid JSON format received."})
            except base64.binascii.Error as b64_error:
                logger.error(f"Base64 decoding error: {b64_error}")
                await websocket.send_json({"error": f"Invalid base64 data: {b64_error}"})
            except FileNotFoundError as fnf_error:
                logger.error(f"File not found during processing: {fnf_error}", exc_info=True)
                await websocket.send_json({"error": f"Server configuration error: {fnf_error}"})
            except RuntimeError as processing_error:
                logger.error(f"Processing error: {processing_error}", exc_info=True)
                await websocket.send_json({"error": f"Video generation failed: {processing_error}"})
            except Exception as e:
                logger.error(f"Unhandled error processing WebSocket message: {str(e)}", exc_info=True)
                await websocket.send_json({"error": f"An unexpected server error occurred: {str(e)}"})

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client.")
    except Exception as e:
        logger.error(f"Unhandled WebSocket exception: {str(e)}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Unhandled server error: {str(e)}")
        except Exception:
            pass 
    finally:
        logger.info("WebSocket endpoint handler finished.")
