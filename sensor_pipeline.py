import threading
import time
from collections import deque
import cv2
import numpy as np

class ThreadSafeBuffer:
    def __init__(self, maxlen=600):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self._counter = 0

    def append(self, item):
        with self.lock:
            self._counter += 1
            # item can be (timestamp, frame) or (id, timestamp, frame)
            if len(item) == 2:
                self.buffer.append((self._counter, item[0], item[1]))
            else:
                self.buffer.append(item)

    def get_snapshot(self):
        with self.lock:
            return list(self.buffer)

    def get_latest(self):
        with self.lock:
            return self.buffer[-1] if self.buffer else None


class SensorDataPipeline:
    def __init__(self, frame_width=1280, frame_height=720, 
                 origin=(25.3176, 82.9739), destination=(25.4358, 81.8463), 
                 trip_duration_seconds=300, video_source=0):
        self.video_buffer = ThreadSafeBuffer(maxlen=600)
        
        # High Definition 720p base resolution for crisp visual clarity
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_source = video_source 
        
        self.origin = origin
        self.destination = destination
        self.trip_duration = trip_duration_seconds
        self.trip_start_time = None
        
        self.running = False
        self.video_thread = None
        
        # State for visual speed estimation (Optical Flow on road)
        self.current_estimated_speed = 0.0
        self.prev_gray = None
        self.prev_pts = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.trip_start_time = time.time()
        
        self.video_thread = threading.Thread(target=self._video_ingest_loop, daemon=True)
        self.video_thread.start()
        print(f"[INFO] High-Definition ADAS sensor pipeline started with source: {self.video_source}")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        print("[INFO] Sensor pipeline stopped cleanly.")

    def _estimate_speed_from_frame(self, frame):
        """
        Calculates real-time ego-speed from video frames by measuring 
        road motion (optical flow) in the lower half of the image.
        Uses a downscaled ROI for ultra-low latency (< 1ms).
        """
        self.flow_skip = getattr(self, "flow_skip", 0) + 1
        if self.flow_skip % 3 != 0 and self.current_estimated_speed > 0:
            return self.current_estimated_speed

        h, w = frame.shape[:2]
        # Road ROI: lower half of the frame
        road_crop = frame[int(h * 0.55):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
        if road_crop.size == 0:
            return self.current_estimated_speed

        # Downscale ROI for fast feature extraction
        small_roi = cv2.resize(road_crop, (240, 120), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_roi, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=40, qualityLevel=0.02, minDistance=6)
            return self.current_estimated_speed

        # Calculate optical flow (Lucas-Kanade)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)

        speed = self.current_estimated_speed
        if next_pts is not None and status is not None:
            good_old = self.prev_pts[status == 1]
            good_new = next_pts[status == 1]

            if len(good_old) > 5:
                displacements = np.linalg.norm(good_new - good_old, axis=1)
                avg_displacement = float(np.mean(displacements))
                
                # Scale factor tuned for normalized speed estimation
                target_speed = avg_displacement * 7.5
                speed = 0.7 * self.current_estimated_speed + 0.3 * target_speed
                if speed < 2.0:
                    speed = 0.0

        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=40, qualityLevel=0.02, minDistance=6)
        self.current_estimated_speed = speed
        return speed

    def _video_ingest_loop(self):
        cap = cv2.VideoCapture(self.video_source)
        
        # If webcam, request 720p HD
        if isinstance(self.video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 60:
            fps = 30.0  
            
        target_interval = 1.0 / fps
        monotonic_counter = 0

        # Calculate aspect-ratio preserving target dimensions
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w > 0 and src_h > 0:
            aspect = src_w / src_h
            if aspect >= (16 / 9):
                target_w = self.frame_width
                target_h = int(round(target_w / aspect))
            else:
                target_h = self.frame_height
                target_w = int(round(target_h * aspect))
            # Keep dimensions even for codec compatibility
            target_w = target_w if target_w % 2 == 0 else target_w - 1
            target_h = target_h if target_h % 2 == 0 else target_h - 1
        else:
            target_w = self.frame_width
            target_h = self.frame_height
        
        while self.running:
            loop_start = time.time()
            ret, frame = cap.read()
            monotonic_counter += 1
            timestamp = self.trip_start_time + (monotonic_counter / fps)
            
            if ret and frame is not None:
                # High-fidelity resize: INTER_AREA downscaling preserves sharp edges from 4K/1080p
                if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                    interp = cv2.INTER_AREA if (frame.shape[1] > target_w) else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
                
                # Compute visual ego-speed
                _ = self._estimate_speed_from_frame(frame)
                
                # Store frame in thread-safe buffer
                self.video_buffer.append((timestamp, frame))
            else:
                if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                    # Loop video seamlessly: rewind video file without breaking monotonic timestamps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.prev_gray = None
                    self.prev_pts = None
                    ret_loop, frame_loop = cap.read()
                    if ret_loop and frame_loop is not None:
                        if (frame_loop.shape[1], frame_loop.shape[0]) != (target_w, target_h):
                            interp = cv2.INTER_AREA if (frame_loop.shape[1] > target_w) else cv2.INTER_LINEAR
                            frame_loop = cv2.resize(frame_loop, (target_w, target_h), interpolation=interp)
                        _ = self._estimate_speed_from_frame(frame_loop)
                        self.video_buffer.append((timestamp, frame_loop))
                else:
                    dummy_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    self.video_buffer.append((timestamp, dummy_frame))
            
            elapsed = time.time() - loop_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
        cap.release()

    def get_synchronized_sample(self, target_timestamp):
        if not self.trip_start_time:
            return None
            
        elapsed_trip = target_timestamp - self.trip_start_time
        progress = min(elapsed_trip / self.trip_duration, 1.0)
        
        current_lat = self.origin[0] + progress * (self.destination[0] - self.origin[0])
        current_lon = self.origin[1] + progress * (self.destination[1] - self.origin[1])
        
        return {
            "timestamp": target_timestamp,
            "imu": {
                "accel": (0.0, 0.0, 9.81 + (np.random.randn() * 0.1)),
                "gyro": (0.0, 0.0, 0.0)
            },
            "gps": {
                "lat": current_lat,
                "lon": current_lon,
                "speed": self.current_estimated_speed
            }
        }