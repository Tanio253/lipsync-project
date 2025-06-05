# Real-Time Lip-Sync WebSocket API

This project implements a real-time, streaming lip-syncing system using a Wav2Lip model.

## Table of Contents
* [System Architecture](#system-architecture)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Running the System](#running-the-system)
* [How to Test](#how-to-test)

## System Architecture
The system is designed using a microservice architecture, composed of a **Gateway**, a **Worker**, and a **Redis** message broker. This separation of concerns ensures scalability and robustness.

Here is the data flow:
1.  **Client (Browser)**: The user accesses a web-based client, selects an image, and initiates a session. Their microphone audio is captured and streamed chunk by chunk over a WebSocket connection.
2.  **Gateway (FastAPI & WebSocket)**:
    * Accepts the WebSocket connection and receives the initial face image.
    * Pushes a `start_session` job, including the image, to the `worker_queue` in Redis.
    * Listens for audio chunks from the client and forwards them as `process_audio` jobs to the same Redis queue.
    * Subscribes to a unique Redis Pub/Sub channel to listen for generated video frames from the worker.
    * Streams the received frames back to the client over the WebSocket, providing a real-time video feed.
3.  **Redis**:
    * Acts as the communication backbone between the gateway and the worker.
    * A **List** (`worker_queue`) is used as a task queue to ensure jobs are processed in order.
    * A **Pub/Sub** channel (`results:{client_id}`) is used to send the processed video frames directly back to the correct gateway instance, enabling real-time, low-latency communication.
4.  **Worker (Wav2Lip Model)**:
    * A dedicated Python process that continuously monitors the `worker_queue` for incoming jobs.
    * **On `start_session`**: It decodes the image, detects the face, and prepares a `Wav2LipStreamingSession` to manage the state.
    * **On `process_audio`**: It adds the incoming audio to a buffer. When enough audio is collected to generate a video frame, it runs the Wav2Lip model inference.
    * The generated lip-synced face is composited back onto the original image.
    * The final frame is encoded and published to the Redis Pub/Sub channel, which the gateway forwards to the client.

This entire system is containerized using Docker, allowing for consistent and straightforward deployment.

## Prerequisites
Before you begin, ensure you have the following installed on your system:
* **Docker**: [https://www.docker.com/get-started](https://www.docker.com/get-started)
* **Docker Compose**: (Usually included with Docker Desktop)

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Tanio253/streaming-Wav2Lip
    cd streaming-Wav2Lip
    ```
    
## Running the System
With the model in place, you can start the entire application using a single Docker Compose command:

```bash
docker-compose up --build
```

## How to Test
The gateway provides a built-in HTML client for easy testing.

1.  **Open the Client**: Once the Docker containers are running, navigate to the following URL in your web browser:
    [http://localhost:8000](http://localhost:8000)

2.  **Start a Session**:
    * Click **"Choose File"** and select a clear, front-facing image (PNG or JPEG).
    * Click the **"Start LipSync Session"** button.

3.  **Stream Audio**:
    * **Allow** the microphone permission.
    * Click the **"Start Recording & Streaming"** button.
    * Begin speaking into your microphone.

