import threading
import time
from collections import deque
import cv2
import numpy as np
import os
import json

class ThreadSafeBuffer:
    """A thread-safe circular ring buffer using collections.deque."""
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            self.buffer.append(item)

    def get_snapshot(self):
        """Returns a list copy of the current buffer contents."""
        with self.lock:
            return list(self.buffer)

class SensorDataPipeline:
    def __init__(self):
        # Buffer capacities based on target rates & 20-second windows:
        # Video: 30 FPS * 20s = 600 frames
        # IMU: 100 Hz * 20s = 2000 readings
        # GPS: 2 Hz * 20s = 40 readings
        self.video_buffer = ThreadSafeBuffer(maxlen=600)
        self.imu_buffer = ThreadSafeBuffer(maxlen=2000)
        self.gps_buffer = ThreadSafeBuffer(maxlen=40)
        
        self.running = False

    def start(self):
        self.running = True
        
        # Start background threads for concurrent ingestion
        self.video_thread = threading.Thread(target=self._video_ingest_loop, daemon=True)
        self.imu_thread = threading.Thread(target=self._imu_ingest_loop, daemon=True)
        self.gps_thread = threading.Thread(target=self._gps_ingest_loop, daemon=True)
        
        self.video_thread.start()
        self.imu_thread.start()
        self.gps_thread.start()
        print("Sensor ingestion pipeline started successfully.")

    def stop(self):
        self.running = False
        print("Stopping sensor ingestion pipeline...")

    def _video_ingest_loop(self):
        # Open default camera (falls back to dummy frame if no webcam is connected)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        while self.running:
            timestamp = time.time()
            ret, frame = cap.read()
            if ret:
                self.video_buffer.append((timestamp, frame))
            else:
                # Fallback synthetic frame if no physical camera is connected
                dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                self.video_buffer.append((timestamp, dummy_frame))
                time.sleep(1/30) # simulate 30fps

        cap.release()

    def _imu_ingest_loop(self):
        target_interval = 1.0 / 100.0  # 100 Hz
        while self.running:
            start_time = time.time()
            timestamp = time.time()
            mock_imu_data = {
                "accel": (np.random.randn(), np.random.randn(), 9.81 + np.random.randn()),
                "gyro": (np.random.randn(), np.random.randn(), np.random.randn())
            }
            self.imu_buffer.append((timestamp, mock_imu_data))
            
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def _gps_ingest_loop(self):
        target_interval = 1.0 / 2.0  # 2 Hz
        while self.running:
            start_time = time.time()
            timestamp = time.time()
            mock_gps_data = {
                "lat": 37.7749 + np.random.randn() * 0.0001,
                "lon": -122.4194 + np.random.randn() * 0.0001,
                "speed": 12.5
            }
            self.gps_buffer.append((timestamp, mock_gps_data))
            
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    def get_synchronized_sample(self, target_timestamp):
        """Interpolates/matches IMU and GPS data to a specific video frame timestamp."""
        imu_snapshot = self.imu_buffer.get_snapshot()
        gps_snapshot = self.gps_buffer.get_snapshot()
        
        if not imu_snapshot or not gps_snapshot:
            return None

        closest_imu = min(imu_snapshot, key=lambda x: abs(x[0] - target_timestamp))
        closest_gps = min(gps_snapshot, key=lambda x: abs(x[0] - target_timestamp))

        return {
            "timestamp": target_timestamp,
            "imu": closest_imu[1],
            "gps": closest_gps[1]
        }


class EvidenceGenerator:
    def __init__(self, pipeline, output_dir="events_output"):
        self.pipeline = pipeline
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_incident_evidence(self, event_type, severity, trigger_timestamp, window_seconds=5):
        start_time = trigger_timestamp - window_seconds
        end_time = trigger_timestamp + window_seconds

        video_snapshot = self.pipeline.video_buffer.get_snapshot()
        clip_frames = [
            (ts, frame) for ts, frame in video_snapshot 
            if start_time <= ts <= end_time
        ]

        if not clip_frames:
            print("[WARNING] No frames found in buffer for the given window.")
            return None

        sync_data = self.pipeline.get_synchronized_sample(trigger_timestamp)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(trigger_timestamp))
        video_filename = f"{event_type}_{timestamp_str}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        json_filename = f"{event_type}_{timestamp_str}.json"
        json_path = os.path.join(self.output_dir, json_filename)

        sample_frame = clip_frames[0][1]
        height, width, _ = sample_frame.shape
        fps = 30.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for _, frame in clip_frames:
            out.write(frame)
        out.release()

        event_metadata = {
            "event_type": event_type,
            "severity_score": severity,
            "trigger_timestamp_utc": trigger_timestamp,
            "gps_location": {
                "latitude": sync_data["gps"]["lat"] if sync_data else None,
                "longitude": sync_data["gps"]["lon"] if sync_data else None,
                "speed": sync_data["gps"]["speed"] if sync_data else None
            },
            "video_clip_path": video_path
        }

        with open(json_path, "w") as f:
            json.dump(event_metadata, f, indent=4)

        print(f"[SUCCESS] Evidence generated! Clip saved to: {video_path}")
        return event_metadata


# --- Main Execution Block ---
if __name__ == "__main__":
    pipeline = SensorDataPipeline()
    pipeline.start()

    try:
        print("Letting buffers populate for 4 seconds...")
        time.sleep(4)
        
        evidence_gen = EvidenceGenerator(pipeline)
        
        current_time = time.time()
        print("\n--- Simulating Harsh Braking Event ---")
        evidence_gen.generate_incident_evidence(
            event_type="Harsh_Braking",
            severity=4,
            trigger_timestamp=current_time,
            window_seconds=5
        )

    finally:
        pipeline.stop()