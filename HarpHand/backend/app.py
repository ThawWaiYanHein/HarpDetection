"""
Harp string detection API: audio (model + video) or hand detection (video + optional weights).
Run: python -m uvicorn app:app --reload
"""

import os
import uuid
import shutil
import subprocess
import csv
import json
import cv2
import numpy as np
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inference import run_pipeline

_backend_dir = Path(__file__).resolve().parent
_project_root = _backend_dir.parent  # used only for FALLBACK_WEIGHTS (best.pt in parent folder)
try:
    from harp_hand_detector import run as run_hand_detector
except ImportError as e:
    import traceback
    print(f"Warning: Could not import harp_hand_detector: {e}")
    print(traceback.format_exc())
    run_hand_detector = None
except Exception as e:
    import traceback
    print(f"Warning: Error loading harp_hand_detector: {e}")
    print(traceback.format_exc())
    run_hand_detector = None

app = FastAPI(title="Harp String Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = _backend_dir / "uploads"
OUTPUT_DIR = _backend_dir / "outputs"
WEIGHTS_DIR = _backend_dir / "weights"
MODELS_DIR = _backend_dir / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default model and weights (user can place default.keras and best.pt here to skip upload)
DEFAULT_MODEL = MODELS_DIR / "default.keras"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
FALLBACK_WEIGHTS = _project_root / "best.pt"

jobs = {}


def create_pluck_filtered_video(
    original_video: str,
    hand_csv: str,
    audio_onsets: list[float],
    output_path: str,
    fps: float = 30.0,
    pluck_window: float = 0.15,
):
    """
    Create video with hand annotations only at pluck moments. Uses frames in the
    window BEFORE each audio onset (finger on string just before pluck), not after.
    """
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {original_video}")
    
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (W, H)
    )
    
    # Load hand events grouped by time
    hand_events_by_time = {}
    if os.path.isfile(hand_csv):
        try:
            with open(hand_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    time_str = row.get("time", "00:00.00")
                    parts = time_str.split(":")
                    if len(parts) == 2:
                        try:
                            minutes = int(parts[0])
                            sec_part = float(parts[1])
                            hand_time = minutes * 60 + sec_part
                            # Round to nearest frame time
                            frame_idx = int(round(hand_time * fps_vid))
                            if frame_idx not in hand_events_by_time:
                                hand_events_by_time[frame_idx] = []
                            hand_events_by_time[frame_idx].append(row)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Warning: Could not load hand CSV: {e}")
    
    # Pluck frames: only BEFORE onset (finger on string just before sound)
    pluck_frames = set()
    for onset_time in audio_onsets:
        frame_start = int(round((onset_time - pluck_window) * fps_vid))
        frame_end = int(round(onset_time * fps_vid))
        for fidx in range(frame_start, frame_end + 1):
            pluck_frames.add(fidx)
    
    # Color palette for strings (16 strings)
    def string_color(sid):
        hue = int(120 * (sid - 1) / 15)
        hsv = np.uint8([[[hue, 180, 240]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr))
    
    fidx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            
            # Only draw hand annotations if this frame is near a pluck
            if fidx in pluck_frames:
                events = hand_events_by_time.get(fidx, [])
                y_offset = 30
                for event in events:
                    string_id = event.get("string", "")
                    finger = event.get("finger", "")
                    dist_px = event.get("dist_px", "")
                    sid = event.get("sid", "")
                    
                    # Draw string label
                    if string_id or sid:
                        try:
                            sid_num = int(sid) if sid else int(string_id.replace("S", "")) if string_id else None
                            if sid_num is None:
                                continue
                            color = string_color(sid_num)
                            label = f"S{sid_num}"
                            if finger:
                                label += f" ({finger})"
                            if dist_px:
                                try:
                                    dist = float(dist_px)
                                    label += f" {dist:.0f}px"
                                except:
                                    pass
                            
                            # Draw text label with background for visibility
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            text_x, text_y = 10, y_offset
                            cv2.rectangle(
                                frame,
                                (text_x - 2, text_y - text_size[1] - 2),
                                (text_x + text_size[0] + 2, text_y + 2),
                                (0, 0, 0), -1
                            )
                            cv2.putText(
                                frame, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2, cv2.LINE_AA
                            )
                            y_offset += 25
                        except (ValueError, AttributeError):
                            continue
            
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def run_job_audio(job_id: str, model_path: str, video_path: str, use_yin_fallback: bool):
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (audio)..."}
        csv_path, video_out_path, df = run_pipeline(
            model_path, video_path, str(OUTPUT_DIR / job_id), use_yin_fallback=use_yin_fallback
        )
        jobs[job_id] = {
            "status": "done",
            "csv_path": csv_path,
            "video_path": video_out_path,
            "rows": len(df),
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


def run_job_hand(job_id: str, video_path: str, weights_path: str | None):
    if run_hand_detector is None:
        jobs[job_id] = {"status": "error", "message": "Hand detector not available (harp_hand_detector not found)."}
        return
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (hand detection)..."}
        out_dir = str(OUTPUT_DIR / job_id)
        csv_path, video_out_path = run_hand_detector(
            video_path,
            output_dir=out_dir,
            preview=False,
            weights_path=weights_path,
        )
        # Re-encode to H.264 so browsers can play it (OpenCV mp4v isn't browser-compatible)
        from inference import FFMPEG_CMD
        h264_path = os.path.join(out_dir, "video_detected_h264.mp4")
        try:
            subprocess.run([
                FFMPEG_CMD, "-y",
                "-i", video_out_path,
                "-i", video_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                h264_path,
            ], check=True, capture_output=True, text=True)
            video_out_path = h264_path
        except Exception as e:
            print(f"Warning: H.264 re-encode failed, using original: {e}")
        rows = 0
        if os.path.isfile(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = max(0, sum(1 for _ in f) - 1)
        jobs[job_id] = {
            "status": "done",
            "csv_path": csv_path,
            "video_path": video_out_path,
            "rows": rows,
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


def run_job_both(
    job_id: str,
    model_path: str,
    video_path: str,
    use_yin_fallback: bool,
    weights_path: str | None,
    original_video_path: str,
):
    out_dir = str(OUTPUT_DIR / job_id)
    try:
        jobs[job_id] = {"status": "running", "message": "Processing (audio)..."}
        csv_audio, video_audio, df = run_pipeline(
            model_path, video_path, out_dir, use_yin_fallback=use_yin_fallback
        )
        audio_result = {"csv_path": csv_audio, "video_path": video_audio, "rows": len(df)}

        if run_hand_detector is None:
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": None,
                "hand_error": "Hand detector not available (install ultralytics, mediapipe, opencv-python and put best.pt in backend/weights/).",
            }
            return
        jobs[job_id] = {"status": "running", "message": "Processing (hand)..."}
        try:
            csv_hand, video_hand = run_hand_detector(
                video_path, output_dir=out_dir, preview=False, weights_path=weights_path
            )
            hand_rows = 0
            if os.path.isfile(csv_hand):
                with open(csv_hand, "r", encoding="utf-8") as f:
                    hand_rows = max(0, sum(1 for _ in f) - 1)
            hand_result = {"csv_path": csv_hand, "video_path": video_hand, "rows": hand_rows}
            
            # Filter hand events to only pluck moments (hand within window BEFORE audio onset)
            jobs[job_id] = {"status": "running", "message": "Filtering hand events to pluck moments..."}
            audio_onsets = df["time_sec"].tolist() if "time_sec" in df.columns else []
            PLUCK_WINDOW = 0.15  # 150ms before onset only (finger on string just before pluck)
            
            filtered_hand_csv = os.path.join(out_dir, "hand_filtered.csv")
            hand_events_at_plucks = []
            if os.path.isfile(csv_hand) and len(audio_onsets) > 0:
                try:
                    with open(csv_hand, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        hand_events = list(reader)
                    
                    for hand_row in hand_events:
                        time_str = hand_row.get("time", "00:00.00")
                        parts = time_str.split(":")
                        if len(parts) == 2:
                            try:
                                minutes = int(parts[0])
                                sec_part = float(parts[1])
                                hand_time = minutes * 60 + sec_part
                                
                                # Only include hand event if it's within window BEFORE an onset
                                for onset_time in audio_onsets:
                                    if onset_time - PLUCK_WINDOW <= hand_time <= onset_time:
                                        hand_events_at_plucks.append(hand_row)
                                        break
                            except ValueError:
                                continue
                    
                    # Write filtered CSV
                    if hand_events_at_plucks:
                        with open(filtered_hand_csv, "w", encoding="utf-8", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=hand_events_at_plucks[0].keys())
                            writer.writeheader()
                            writer.writerows(hand_events_at_plucks)
                        hand_result["filtered_csv_path"] = filtered_hand_csv
                        hand_result["filtered_rows"] = len(hand_events_at_plucks)
                except Exception as e:
                    print(f"Warning: Could not filter hand events: {e}")
            
            # Create combined video: hand/string at plucks + original audio + SRT subtitles (top)
            jobs[job_id] = {"status": "running", "message": "Combining results..."}
            combined_video = os.path.join(out_dir, "video_combined.mp4")
            ass_path = os.path.join(out_dir, "overlay.ass")
            srt_path = os.path.join(out_dir, "overlay.srt")
            combined_result = None
            combined_error = None
            
            if not os.path.isfile(video_hand):
                combined_error = f"Hand video not found: {video_hand}"
            else:
                # Create video with hand annotations only at pluck moments (labels already at top)
                jobs[job_id] = {"status": "running", "message": "Creating pluck-filtered video..."}
                pluck_filtered_video = os.path.join(out_dir, "hand_plucks_only.mp4")
                try:
                    cap = cv2.VideoCapture(original_video_path)
                    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()
                    create_pluck_filtered_video(
                        original_video_path, csv_hand, audio_onsets, pluck_filtered_video, fps_vid
                    )
                except Exception as e:
                    print(f"Warning: Could not create pluck-filtered video: {e}")
                    pluck_filtered_video = video_hand  # Fallback to full hand video
                
                # Burn subtitles: prefer ASS (fixed position), else SRT
                from inference import FFMPEG_CMD
                try:
                    if os.path.isfile(ass_path):
                        overlay_abs = os.path.abspath(ass_path).replace("\\", "/")
                        if os.name == "nt":
                            overlay_abs = overlay_abs.replace(":", "\\:", 1)
                        vf = f"subtitles='{overlay_abs}'"
                        result = subprocess.run([
                            FFMPEG_CMD, "-y",
                            "-i", pluck_filtered_video,
                            "-i", original_video_path,
                            "-vf", vf,
                            "-c:v", "libx264",
                            "-c:a", "aac",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-shortest",
                            combined_video
                        ], check=True, capture_output=True, text=True)
                    elif os.path.isfile(srt_path):
                        srt_abs = os.path.abspath(srt_path).replace("\\", "/")
                        if os.name == "nt":
                            srt_abs = srt_abs.replace(":", "\\:", 1)
                        result = subprocess.run([
                            FFMPEG_CMD, "-y",
                            "-i", pluck_filtered_video,
                            "-i", original_video_path,
                            "-vf", f"subtitles='{srt_abs}':force_style='Fontsize=28,BorderStyle=1,Outline=2,Shadow=1,Alignment=7,MarginV=32,MarginR=32'",
                            "-c:v", "libx264",
                            "-c:a", "aac",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-shortest",
                            combined_video
                        ], check=True, capture_output=True, text=True)
                    else:
                        # No SRT: just mux pluck video + audio
                        result = subprocess.run([
                            FFMPEG_CMD, "-y",
                            "-i", pluck_filtered_video,
                            "-i", original_video_path,
                            "-c:v", "libx264",
                            "-c:a", "aac",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-shortest",
                            combined_video
                        ], check=True, capture_output=True, text=True)
                    if os.path.isfile(combined_video):
                        combined_result = {"video_path": combined_video}
                    else:
                        combined_error = "Combined video file was not created"
                except subprocess.CalledProcessError as e:
                    combined_error = f"FFmpeg error: {e.stderr[:200] if e.stderr else str(e)}"
                except Exception as e:
                    combined_error = f"Error creating combined video: {str(e)}"
            
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": hand_result,
                "combined": combined_result,
                "combined_error": combined_error,
            }
        except Exception as hand_err:
            jobs[job_id] = {
                "status": "done",
                "audio": audio_result,
                "hand": None,
                "hand_error": str(hand_err),
            }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


@app.post("/api/upload")
async def upload_and_run(
    background_tasks: BackgroundTasks,
    method: str = Form("audio"),
    use_default_model: str = Form("false"),
    model: UploadFile = File(None),
    video: UploadFile = File(...),
    mode: str = Form("hybrid"),
    use_default_weights: str = Form("false"),
    weights: UploadFile = File(None),
):
    if not video.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        raise HTTPException(400, "Video must be .mp4, .mov, .mkv, .avi, or .webm")

    method = method.lower()
    if method not in ("audio", "hand", "both"):
        raise HTTPException(400, "method must be 'audio', 'hand', or 'both'")

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    (OUTPUT_DIR / job_id).mkdir(parents=True, exist_ok=True)

    video_path = job_dir / (video.filename or "video.mp4")
    try:
        with open(video_path, "wb") as f:
            f.write(await video.read())
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Save video failed: {e}")

    if method == "audio":
        use_default = use_default_model.lower() == "true"
        if use_default and DEFAULT_MODEL.exists():
            model_path = str(DEFAULT_MODEL)
        elif model and model.filename and model.filename.lower().endswith(".keras"):
            model_path = job_dir / (model.filename or "model.keras")
            try:
                with open(model_path, "wb") as f:
                    f.write(await model.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save model failed: {e}")
            model_path = str(model_path)
        else:
            raise HTTPException(
                400,
                "For audio detection, upload a .keras model file or place default.keras in backend/models/ and use 'Use default model'.",
            )
        use_yin = mode.lower() == "hybrid"
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(run_job_audio, job_id, model_path, str(video_path), use_yin)
    elif method == "both":
        use_default = use_default_model.lower() == "true"
        if use_default and DEFAULT_MODEL.exists():
            model_path = str(DEFAULT_MODEL)
        elif model and model.filename and model.filename.lower().endswith(".keras"):
            model_path = job_dir / (model.filename or "model.keras")
            try:
                with open(model_path, "wb") as f:
                    f.write(await model.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save model failed: {e}")
            model_path = str(model_path)
        else:
            raise HTTPException(
                400,
                "For both, upload a .keras model file or place default.keras in backend/models/ and use 'Use default model'.",
            )
        use_yin = mode.lower() == "hybrid"
        weights_path = None
        use_default_w = use_default_weights.lower() == "true"
        if weights and weights.filename and weights.filename.lower().endswith(".pt"):
            wpath = job_dir / (weights.filename or "weights.pt")
            try:
                with open(wpath, "wb") as f:
                    f.write(await weights.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save weights failed: {e}")
            weights_path = str(wpath)
        elif use_default_w or not (weights and weights.filename):
            if DEFAULT_WEIGHTS.exists():
                weights_path = str(DEFAULT_WEIGHTS)
            elif FALLBACK_WEIGHTS.exists():
                weights_path = str(FALLBACK_WEIGHTS)
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(
            run_job_both, job_id, model_path, str(video_path), use_yin, weights_path, str(video_path)
        )
    else:
        weights_path = None
        if weights and weights.filename and weights.filename.lower().endswith(".pt"):
            weights_path = job_dir / (weights.filename or "weights.pt")
            try:
                with open(weights_path, "wb") as f:
                    f.write(await weights.read())
            except Exception as e:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(500, f"Save weights failed: {e}")
            weights_path = str(weights_path)
        else:
            if DEFAULT_WEIGHTS.exists():
                weights_path = str(DEFAULT_WEIGHTS)
            elif FALLBACK_WEIGHTS.exists():
                weights_path = str(FALLBACK_WEIGHTS)
            else:
                raise HTTPException(
                    400,
                    "For hand detection, upload a .pt weights file or place best.pt in backend/weights/ or project root.",
                )
        jobs[job_id] = {"status": "queued"}
        background_tasks.add_task(run_job_hand, job_id, str(video_path), weights_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


def _get_result_path(job: dict, kind: str, type_key: str | None) -> tuple[str, str]:
    """Return (path, filename). type_key is 'audio', 'hand', or 'combined' for both-mode jobs."""
    if "audio" in job:
        if type_key == "combined":
            combined = job.get("combined")
            if not combined:
                raise HTTPException(404, "No combined video available")
            path = combined.get("video_path")
            name = "harp_combined.mp4"
        elif not type_key:
            raise HTTPException(400, "For this job, use ?type=audio, ?type=hand, or ?type=combined")
        else:
            part = job.get(type_key)
            if not part:
                raise HTTPException(404, f"No {type_key} result")
            path = part.get("csv_path" if kind == "csv" else "video_path")
            name = f"harp_{type_key}_{'predictions' if kind == 'csv' else 'video'}.{'csv' if kind == 'csv' else 'mp4'}"
    else:
        path = job.get("csv_path" if kind == "csv" else "video_path")
        name = "harp_predictions.csv" if kind == "csv" else "harp_labeled.mp4"
    return (path, name)


@app.get("/api/download/csv/{job_id}")
def download_csv(job_id: str, type: str = Query(None, alias="type")):
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, filename = _get_result_path(job, "csv", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "CSV not found")
    return FileResponse(path, filename=filename, media_type="text/csv")


@app.get("/api/download/video/{job_id}")
def download_video(job_id: str, type: str = Query(None, alias="type")):
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, filename = _get_result_path(job, "video", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.get("/api/video-url/{job_id}")
def get_video_url(job_id: str, type: str = Query(None, alias="type")):
    """Get video URL for preview (returns URL path, not file download)."""
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, _ = _get_result_path(job, "video", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Video not found")
    # Return relative path that frontend can use
    rel_path = os.path.relpath(path, _backend_dir).replace("\\", "/")
    return {"url": f"/api/video-stream/{job_id}?type={type or ''}"}


@app.get("/api/video-stream/{job_id}")
def stream_video(job_id: str, type: str = Query(None, alias="type")):
    """Stream video for preview (supports range requests for seeking)."""
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    path, filename = _get_result_path(job, "video", type)
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Video not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.get("/api/logs/{job_id}")
def get_logs(job_id: str):
    """Get combined log events from audio and hand detection CSVs. Hand events filtered to pluck moments only."""
    if job_id not in jobs or jobs[job_id].get("status") != "done":
        raise HTTPException(404, "Job not ready or not found")
    job = jobs[job_id]
    events = []
    audio_onsets = []
    PLUCK_WINDOW = 0.15  # Only consider hand 0–150ms BEFORE onset (finger on string)
    TRACKING_WINDOW = 0.5  # Window to determine if hand event is "tracking" vs "detected"
    
    # Parse audio CSV and collect onsets
    if "audio" in job and job["audio"]:
        audio_csv = job["audio"].get("csv_path")
        if audio_csv and os.path.isfile(audio_csv):
            try:
                with open(audio_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            time_sec = float(row.get("time_sec", 0))
                            audio_onsets.append(time_sec)
                            strings = row.get("predicted_strings", "").strip()
                            if strings:
                                for s in strings.split(","):
                                    s = s.strip()
                                    if s and s.isdigit():
                                        s_num = int(s)
                                        prob_key = f"prob_S{s_num}"
                                        prob = float(row.get(prob_key, 0)) if prob_key in row else 0.0
                                        events.append({
                                            "time": time_sec,
                                            "type": "audio",
                                            "string": f"S{s}",
                                            "confidence": prob,
                                            "method": row.get("used", "model"),
                                        })
                        except (ValueError, KeyError) as e:
                            print(f"Skipping invalid audio CSV row: {e}")
                            continue
            except Exception as e:
                print(f"Error parsing audio CSV: {e}")
    
    # Parse hand CSV: only one hand+string entry per pluck (sound event)
    # Group hand events by nearest audio onset; emit at most one hand row per pluck
    if "hand" in job and job["hand"]:
        hand_csv = job["hand"].get("csv_path")
        if hand_csv and os.path.isfile(hand_csv) and len(audio_onsets) > 0:
            try:
                # Collect all hand events with time
                hand_rows = []
                with open(hand_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        time_str = row.get("time", "00:00.00")
                        parts = time_str.split(":")
                        if len(parts) != 2:
                            continue
                        try:
                            minutes = int(parts[0])
                            sec_part = float(parts[1])
                            hand_time = minutes * 60 + sec_part
                            dist_px = float(row.get("dist_px", 0)) if row.get("dist_px") else 0.0
                            max_dist = 20.0
                            confidence = max(0.0, min(1.0, 1.0 - (dist_px / max_dist))) if dist_px > 0 else 0.5
                            hand_rows.append({
                                "time": hand_time,
                                "string": row.get("string", "?"),
                                "finger": row.get("finger", ""),
                                "distance": dist_px,
                                "confidence": confidence,
                            })
                        except (ValueError, KeyError):
                            continue
                
                # One hand event per sound: only hand events BEFORE onset (finger on string)
                unique_onsets = sorted(set(audio_onsets))
                for onset_time in unique_onsets:
                    n_strings = sum(1 for e in events if e.get("type") == "audio" and e.get("time") == onset_time)
                    if n_strings <= 0:
                        continue
                    audio_strings_at_onset = {e["string"] for e in events if e.get("type") == "audio" and e.get("time") == onset_time}
                    # Hand must be in [onset - PLUCK_WINDOW, onset]; rank by how close to onset (before)
                    matches = [(onset_time - h["time"], h) for h in hand_rows if (onset_time - PLUCK_WINDOW <= h["time"] <= onset_time)]
                    best_by_key = {}
                    for dt, h in matches:
                        s = str(h["string"]).strip()
                        string_display = s if (s and s.upper().startswith("S")) else (f"S{s}" if s else "S?")
                        key = (string_display, h["finger"])
                        if key not in best_by_key or dt < best_by_key[key][0]:
                            best_by_key[key] = (dt, h, string_display)
                    candidates = list(best_by_key.values())
                    # Prefer hand events whose string matches an audio string at this pluck
                    matching = [c for c in candidates if c[2] in audio_strings_at_onset]
                    if matching:
                        candidates = matching
                    if not candidates:
                        continue
                    # Pick exactly n_strings hand events: 1 sound -> best finger; 2 sounds -> best thumb + best index
                    if n_strings == 1:
                        chosen = [max(candidates, key=lambda c: c[1]["confidence"])]
                    else:
                        by_finger = {}
                        for (dt, h, string_display) in candidates:
                            by_finger.setdefault(h["finger"], []).append((dt, h, string_display))
                        best_thumb = max(by_finger.get("thumb", []), key=lambda c: c[1]["confidence"], default=None)
                        best_index = max(by_finger.get("index", []), key=lambda c: c[1]["confidence"], default=None)
                        if best_thumb and best_index:
                            chosen = [best_thumb, best_index]
                        else:
                            chosen = sorted(candidates, key=lambda c: -c[1]["confidence"])[:n_strings]
                    for (dt, h, string_display) in chosen:
                        events.append({
                            "time": onset_time,
                            "type": "hand",
                            "string": string_display,
                            "finger": h["finger"],
                            "distance": h["distance"],
                            "frame": 0,
                            "status": "detected",
                            "confidence": h["confidence"],
                        })
            except Exception as e:
                print(f"Error parsing hand CSV: {e}")
        elif hand_csv and os.path.isfile(hand_csv) and len(audio_onsets) == 0:
            # Hand-only job: include all hand events (no pluck filter)
            try:
                with open(hand_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        time_str = row.get("time", "00:00.00")
                        parts = time_str.split(":")
                        if len(parts) == 2:
                            try:
                                hand_time = int(parts[0]) * 60 + float(parts[1])  # MM:SS.mm
                                dist_px = float(row.get("dist_px", 0)) if row.get("dist_px") else 0.0
                                confidence = max(0.0, min(1.0, 1.0 - (dist_px / 20.0))) if dist_px > 0 else 0.5
                                events.append({
                                    "time": hand_time,
                                    "type": "hand",
                                    "string": f"S{row.get('string', '?')}",
                                    "finger": row.get("finger", ""),
                                    "distance": dist_px,
                                    "frame": 0,
                                    "status": "detected",
                                    "confidence": confidence,
                                })
                            except (ValueError, KeyError):
                                continue
            except Exception as e:
                print(f"Error parsing hand CSV: {e}")
    
    # Sort by time
    events.sort(key=lambda x: x["time"])
    
    # Add sequential entry numbers (0001, 0002, etc.)
    for idx, event in enumerate(events, start=1):
        event["entry_number"] = f"{idx:04d}"
    
    return {"events": events}


@app.get("/api/defaults")
def get_defaults():
    """Report whether bundled default model and weights exist (so frontend can show 'Use default' options)."""
    return {
        "default_model": DEFAULT_MODEL.exists(),
        "default_weights": DEFAULT_WEIGHTS.exists() or FALLBACK_WEIGHTS.exists(),
    }


@app.get("/")
def root():
    return {"message": "Harp String Detection API", "docs": "/docs"}
