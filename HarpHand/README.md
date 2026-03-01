# HarpHand — Harp String Detection Using Audio Classification and Computer Vision

A web-based system for detecting which strings are plucked on a 16-string harp by combining **audio classification** (deep learning) with **hand position tracking** (computer vision). The system processes video recordings of harp performances and produces per-pluck string predictions, annotated video output, and a generated musical notation grid.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Audio Detection Pipeline](#audio-detection-pipeline)
- [Hand Detection Pipeline](#hand-detection-pipeline)
- [Combined Mode](#combined-mode)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Limitations](#limitations)

---

## Overview

Identifying which string is being played on a harp is non-trivial due to the close frequency spacing of adjacent strings and the rapid plucking motion of the fingers. This project tackles the problem from two complementary angles:

1. **Audio-based classification** — A Keras CNN trained on mel-spectrogram features classifies short audio clips around detected onsets into one of 16 string classes. A YIN pitch estimator serves as a fallback when model confidence is low.

2. **Vision-based hand tracking** — A YOLOv8 object detector localises the 16 harp strings in each video frame while MediaPipe Hand Landmarker tracks fingertip positions. A string is marked as "touched" when a fingertip (thumb or index) falls within a pixel-distance threshold of a string's centerline.

3. **Combined mode** — Audio onsets define *when* a pluck occurs; hand events within a short temporal window before each onset determine *which finger* plucked *which string*, allowing cross-validation of both modalities.

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   React Frontend                      │
│  Upload video + models → View results, logs, video    │
│  Generated Note grid → PDF export                     │
└──────────────┬───────────────────────────┬────────────┘
               │  REST API (FastAPI)       │
┌──────────────▼───────────────────────────▼────────────┐
│                   Python Backend                       │
│                                                        │
│  ┌─────────────────┐      ┌──────────────────────┐    │
│  │ Audio Pipeline   │      │ Hand Pipeline         │    │
│  │ (inference.py)   │      │ (harp_hand_detector)  │    │
│  │                  │      │                       │    │
│  │ FFmpeg → WAV     │      │ YOLOv8 string det.   │    │
│  │ Librosa onset    │      │ MediaPipe hand land.  │    │
│  │ Keras CNN        │      │ Fingertip–string      │    │
│  │ YIN fallback     │      │  distance matching    │    │
│  │ ASS subtitles    │      │ CSV + annotated video │    │
│  └─────────────────┘      └──────────────────────┘    │
│                                                        │
│  Combined mode: pluck-window filtering + video merge   │
└────────────────────────────────────────────────────────┘
```

---

## Audio Detection Pipeline

**File:** `backend/inference.py`

1. **Audio extraction** — FFmpeg extracts a 16 kHz mono WAV from the input video.
2. **Onset detection** — Librosa's `onset_detect` identifies pluck times in the audio signal.
3. **Feature extraction** — For each onset, an 0.8 s clip is converted to:
   - A 128-bin mel-spectrogram (log-scaled, normalised).
   - A 16-dimensional string energy vector (harmonic energy per string fundamental, up to 5 harmonics).
4. **CNN classification** — A trained Keras model (`default.keras`) takes [mel-spectrogram, energy vector] and outputs a 16-class probability distribution.
5. **Hybrid thresholding** — Per-string confidence thresholds are applied. If the top prediction falls below the threshold, a YIN pitch estimator provides a fallback prediction (labelled "String*" in the output).
6. **Subtitle generation** — An ASS subtitle file with fixed top-right positioning (`\pos`, `\an9`) is generated and burned into the video using FFmpeg's `subtitles` filter.

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `SAMPLE_RATE` | 16,000 Hz | Audio resampling rate |
| `CLIP_SEC` | 0.8 s | Duration of each onset clip |
| `N_MELS` | 128 | Mel-spectrogram frequency bins |
| `THR_DEFAULT` | 0.25 | Base confidence threshold |
| `TOP2_DISPLAY_THR` | 0.20 | Threshold for showing secondary string prediction |
| `YIN_WIN_SEC` | 0.15 s | YIN analysis window |

---

## Hand Detection Pipeline

**File:** `backend/harp_hand_detector.py`

1. **String localisation** — YOLOv8 (custom-trained weights `best.pt`, 640×640 input) detects the 16 harp string bounding boxes. Non-maximum suppression (IoU 0.40) removes duplicates. A `StringModel` class maintains a running spatial model of string positions across frames using exponential smoothing.
2. **Hand landmark detection** — MediaPipe Hand Landmarker (float16, up to 2 hands) extracts 21 landmarks per hand per frame.
3. **Touch detection** — For each fingertip (thumb = landmark 4, index = landmark 8), the perpendicular distance to every string's centerline is computed. If the distance falls below `TOUCH_DIST_PX` (20 px) for at least `TOUCH_CONSEC` consecutive frames, a touch event is logged.
4. **Orientation-agnostic geometry** — String centerlines are derived from YOLO bounding box corners, supporting horizontal, vertical, and diagonal string orientations.
5. **Output** — An annotated video with bounding boxes, hand skeletons, and touch labels, plus a CSV log of all touch events.

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `IMGSZ` | 640 | YOLO input resolution |
| `CONF` | 0.08 | Minimum YOLO confidence |
| `TOUCH_DIST_PX` | 20 px | Max fingertip-to-string distance for touch |
| `TOUCH_CONSEC` | 1 | Consecutive frames required to confirm touch |
| `FRAME_SKIP` | 2 | Run YOLO every Nth frame (MediaPipe runs every frame) |

---

## Combined Mode

When both pipelines run together:

1. Audio pipeline runs first to identify pluck times (onsets) and predicted strings.
2. Hand pipeline processes the full video independently.
3. **Pluck-window filtering** — Only hand events within a 150 ms window *before* each audio onset are retained (`onset - 0.15s` to `onset`). This captures the finger position just before the string sounds, accounting for the fact that the hand moves away immediately after plucking.
4. **Cross-validation** — For each pluck, hand events whose detected string matches an audio-predicted string are prioritised. If the audio detects two strings simultaneously, the system selects the best thumb and best index finger events.
5. **Combined video** — A pluck-filtered video (hand annotations only at pluck moments) is overlaid with ASS subtitles and muxed with the original audio using FFmpeg.

---

## Features

- **Three detection modes**: Audio-only, Hand-only, or Both (combined).
- **Default models**: Pre-bundled `.keras` and `.pt` models — users can choose "Use default" or upload their own.
- **Detection log**: Timestamped list/grid view of all detection events with match indicators.
- **Click-to-seek**: Click any log entry or generated note cell to jump the video to that timestamp.
- **Generated Note grid**: 8-column notation display — single strings as numbers, simultaneous plucks underlined, thumb plucks marked with a dot above. Clickable cells and PDF export with multi-page support.
- **Video preview**: Browser-playable H.264 output with subtitle overlays.
- **Navigation**: Next/Previous pluck buttons and a visual timeline strip with pluck markers.
- **Analysis panel**: Per-string match rates and audio vs. hand agreement matrix.
- **CSV download**: Export the full detection log as CSV.
- **PDF download**: Export the generated note grid as a multi-page PDF.

---

## Project Structure

```
HarpHand/
├── backend/
│   ├── app.py                   # FastAPI application and API endpoints
│   ├── inference.py             # Audio detection pipeline (Keras + YIN + FFmpeg)
│   ├── harp_hand_detector.py    # Hand detection pipeline (YOLO + MediaPipe)
│   ├── hand_landmarker.task     # MediaPipe hand model (auto-downloaded if missing)
│   ├── requirements.txt         # Python dependencies
│   ├── get_ffmpeg.ps1           # Helper script to install FFmpeg on Windows
│   ├── models/
│   │   └── default.keras        # Default audio classification model
│   └── weights/
│       └── best.pt              # Default YOLO string detection weights
├── frontend/
│   ├── index.html               # Entry HTML
│   ├── package.json             # Node.js dependencies
│   ├── vite.config.js           # Vite build configuration with API proxy
│   └── src/
│       ├── App.jsx              # Main React application component
│       ├── App.css              # Component styles
│       ├── index.css            # Global styles and theme variables
│       └── main.jsx             # React entry point
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **FFmpeg** installed and available on PATH
- **CUDA** (optional) — GPU acceleration for YOLO and TensorFlow

---

## Installation

### Backend

```bash
cd HarpHand/backend
pip install -r requirements.txt
```

If `hand_landmarker.task` is missing, it will be automatically downloaded from Google's model repository on first run.

### Frontend

```bash
cd HarpHand/frontend
npm install
```

---

## Usage

### Start the backend

```bash
cd HarpHand/backend
python -m uvicorn app:app --reload --port 8000
```

### Start the frontend (development)

```bash
cd HarpHand/frontend
npm run dev
```

The frontend runs on `http://localhost:5173` and proxies API requests to the backend on port 8000.

### Workflow

1. Open `http://localhost:5173` in a browser.
2. Select a detection mode: **Audio**, **Hand**, or **Both**.
3. Choose to use the default bundled models or upload your own `.keras` / `.pt` files.
4. Upload a video file (.mp4, .mov, .mkv, .avi, or .webm).
5. Click **Upload & Detect** and wait for processing to complete.
6. Review results: annotated video, detection log, generated note grid, and per-string analysis.
7. Download outputs: CSV log, annotated video, or generated note PDF.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload video and models, start processing |
| `GET` | `/api/status/{job_id}` | Poll job status |
| `GET` | `/api/logs/{job_id}` | Get detection events (combined log) |
| `GET` | `/api/video-stream/{job_id}` | Stream annotated video |
| `GET` | `/api/download/csv/{job_id}` | Download predictions CSV |
| `GET` | `/api/download/video/{job_id}` | Download annotated video |
| `GET` | `/api/defaults` | Check if default models are available |

---

## Configuration

### Audio pipeline (`inference.py`)

- `THR_ARRAY` — Per-string confidence thresholds (adjust for strings with frequent misclassification).
- `TOP2_DISPLAY_THR` — Minimum confidence to include a secondary string prediction.
- `FLASH_DURATION` — How long each subtitle label stays on screen (seconds).

### Hand pipeline (`harp_hand_detector.py`)

- `IMGSZ` — YOLO input image size (higher = more accurate but slower).
- `FRAME_SKIP` — Process YOLO every Nth frame (1 = every frame; 2 = faster with minimal accuracy loss).
- `TOUCH_DIST_PX` — Maximum pixel distance for a fingertip to be considered touching a string.
- `CONF` — Minimum YOLO detection confidence.

### Combined mode (`app.py`)

- `PLUCK_WINDOW` — Temporal window (seconds) before each audio onset to search for hand events.

---

## Limitations

- The audio model is trained on a specific 16-string harp tuning. Different tunings or string counts require retraining.
- Hand detection accuracy depends on camera angle and lighting. Fingers must be clearly visible.
- The pluck-window approach assumes audio and video are synchronised. Significant A/V drift will degrade combined-mode accuracy.
- Processing speed is bounded by YOLO inference; GPU (CUDA) is recommended for videos longer than 30 seconds.
- Browser video playback requires H.264 encoding, which is handled automatically via FFmpeg.
