import os
import cv2
import threading
import json
import numpy as np
from datetime import datetime
from database import insert_incident


def _async_write_video_clip(video_path, frames, width, height, fps=30.0):
    """
    Encodes and saves a high-quality, browser-compatible H.264 MP4 evidence clip
    with +faststart in a background worker thread. Web browsers load and stream it immediately.
    """
    if not frames:
        return
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        w = width if width % 2 == 0 else width - 1
        h = height if height % 2 == 0 else height - 1
        
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-movflags', '+faststart',
            video_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in frames:
            if (f.shape[1], f.shape[0]) != (w, h):
                f = cv2.resize(f, (w, h))
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait(timeout=10)
        print(f"[EvidenceGenerator] Successfully encoded faststart H.264 clip: {video_path}")
    except Exception as e:
        print(f"[EvidenceGenerator] faststart ffmpeg notice: {e}, using cv2 fallback")
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
        except Exception as ex:
            print(f"[EvidenceGenerator] Video writer fallback error: {ex}")


class EvidenceGenerator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        os.makedirs("events_output", exist_ok=True)

    def generate_incident_evidence(self, *args, **kwargs):
        """
        Generates incident evidence asynchronously without blocking the live video stream.
        Saves trigger snapshot immediately and encodes the MP4 clip in a background worker.
        """
        window_seconds = kwargs.get("window_seconds", 8)
        description = "Hazard event"
        frame = None

        if len(args) == 3 and isinstance(args[2], dict) and "event_type" in args[2]:
            frame, sync_data, event = args
            event_type = event.get("event_type", "Collision_Risk")
            severity = str(event.get("severity", "Medium"))
            trigger_timestamp = event.get("timestamp") or (sync_data.get("timestamp") if sync_data else datetime.now().timestamp())
            description = event.get("description", "Safety incident")
            gps_data = sync_data.get("gps", {}) if sync_data else {}
        elif len(args) >= 4:
            event_type = args[0]
            severity = str(args[1])
            trigger_timestamp = args[2]
            gps_data = args[3] if isinstance(args[3], dict) else {}
            if len(args) >= 5:
                window_seconds = args[4]
            description = kwargs.get("description", f"Incident: {event_type}")
        else:
            event_type = kwargs.get("event_type", "Incident")
            severity = str(kwargs.get("severity", "Medium"))
            trigger_timestamp = kwargs.get("trigger_timestamp", datetime.now().timestamp())
            gps_data = kwargs.get("gps_data", {})
            description = kwargs.get("description", "Incident detected")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{event_type}_{timestamp_str}"
        video_path = os.path.join("events_output", f"{base_filename}.mp4")
        image_path = os.path.join("events_output", f"{base_filename}.jpg")

        # 1. Save instantaneous snapshot image for instant UI display
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            try:
                cv2.imwrite(image_path, frame)
            except Exception:
                pass

        # 2. Extract recent frames from buffer (supports both 2-tuple and 3-tuple buffer items)
        window_frames = []
        if hasattr(self.pipeline, "video_buffer"):
            with self.pipeline.video_buffer.lock:
                buf = self.pipeline.video_buffer.buffer
                for item in buf:
                    if len(item) == 3:
                        _, t, f = item
                    elif len(item) == 2:
                        t, f = item
                    else:
                        continue
                    if (trigger_timestamp - window_seconds) <= t <= (trigger_timestamp + 1.0):
                        window_frames.append(f)

        if window_frames:
            h, w, _ = window_frames[0].shape
        else:
            h, w = 720, 1280
            dummy = frame if (frame is not None and frame.size > 0) else np.zeros((h, w, 3), dtype=np.uint8)
            window_frames = [dummy for _ in range(15)]

        # 3. Offload high-quality H.264 MP4 encoding to background thread
        threading.Thread(
            target=_async_write_video_clip,
            args=(video_path, list(window_frames), w, h, 30.0),
            daemon=True
        ).start()

        # 4. Package incident metadata & insert into SQLite immediately
        incident_metadata = {
            "event_type": event_type,
            "severity_score": severity,
            "description": description,
            "trigger_timestamp_utc": trigger_timestamp,
            "gps_location": {
                "latitude": gps_data.get("lat"),
                "longitude": gps_data.get("lon"),
                "speed": gps_data.get("speed", 0.0)
            },
            "video_clip_path": video_path,
            "review_status": "unreviewed"
        }

        json_path = os.path.join("events_output", f"{base_filename}.json")
        try:
            with open(json_path, "w") as jf:
                json.dump(incident_metadata, jf, indent=4)
        except Exception:
            pass

        insert_incident(trigger_timestamp, event_type, severity, description, video_path)
        print(f"[EvidenceGenerator] Saved incident: {event_type} | H.264 video encoding in background")