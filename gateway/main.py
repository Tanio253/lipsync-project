import asyncio
import json
import logging
import os
import uuid

import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379

app = FastAPI(title="LipSync WebSocket Gateway")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, auto_close_connection_pool=False)

html_test_page_streaming ="""
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
        const BUFFER_SIZE = 4096; // Audio buffer size for ScriptProcessorNode

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

async def redis_results_listener(websocket: WebSocket, client_id: str):
    """Listens to a Redis Pub/Sub channel and forwards messages to the WebSocket client."""
    pubsub = redis_client.pubsub()
    results_channel = f"results:{client_id}"
    await pubsub.subscribe(results_channel)
    logger.info(f"Subscribed to Redis results channel: {results_channel}")
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "message":
                frame_data = json.loads(message["data"])
                await websocket.send_json(frame_data)
    except asyncio.CancelledError:
        logger.info(f"Listener for {client_id} cancelled.")
    except Exception as e:
        logger.error(f"Error in Redis listener for {client_id}: {e}", exc_info=True)
    finally:
        await pubsub.unsubscribe(results_channel)
        logger.info(f"Unsubscribed from {results_channel}")


@app.websocket("/ws/lipsync_stream")
async def websocket_streaming_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connection accepted from {client_id}")

    listener_task = asyncio.create_task(redis_results_listener(websocket, client_id))
    session_id = None

    try:
        while True:
            raw_data = await websocket.receive_text()
            message = json.loads(raw_data)
            message_type = message.get("type")

            if message_type == "start_session":
                session_id = str(uuid.uuid4())
                image_b64 = message.get("image_base64")
                job_payload = {
                    "type": "start_session",
                    "client_id": client_id,
                    "session_id": session_id,
                    "image_b64": image_b64
                }
                await redis_client.rpush("worker_queue", json.dumps(job_payload))
                logger.info(f"Pushed 'start_session' job for session {session_id} to worker queue.")

            elif message_type == "audio_chunk":
                if not session_id:
                    await websocket.send_json({"error": "Session not started."})
                    continue

                audio_chunk_b64 = message.get("audio_chunk_base64")
                job_payload = {
                    "type": "process_audio",
                    "client_id": client_id, 
                    "session_id": session_id,
                    "audio_chunk_b64": audio_chunk_b64
                }
                await redis_client.rpush("worker_queue", json.dumps(job_payload))

            else:
                logger.warning(f"Unknown message type received: {message_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket {client_id} disconnected.")
    except Exception as e:
        logger.error(f"Unhandled WebSocket error for {client_id}: {e}", exc_info=True)
    finally:
        listener_task.cancel()
        if session_id:
            cleanup_payload = {"type": "cleanup_session", "session_id": session_id}
            await redis_client.rpush("worker_queue", json.dumps(cleanup_payload))
            logger.info(f"Pushed 'cleanup_session' job for session {session_id}.")
        await listener_task
        logger.info(f"Streaming WebSocket handler for {client_id} finished.")