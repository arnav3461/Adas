import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class FrontVehicleSpeedEstimator:
    def __init__(self, smoothing_window=10):
        """
        Estimates the speed of the front vehicle using bounding box scale expansion rate.
        This works universally across any video dataset because object growth rate 
        in perspective projection directly correlates with relative speed.
        """
        self.history = defaultdict(list)  # track_id -> [(timestamp, box_height), ...]
        self.smoothing_window = smoothing_window

    def update_and_calculate(self, track_id, bbox, timestamp, ego_speed_kmh=50.0):
        """
        Calculates speed using the rate of change of bounding box height (dh/dt).
        """
        x1, y1, x2, y2 = bbox
        box_height = max(1.0, y2 - y1)
        
        track_history = self.history[track_id]
        track_history.append((timestamp, box_height))
        
        if len(track_history) > self.smoothing_window:
            track_history.pop(0)
            
        if len(track_history) < 3:
            return ego_speed_kmh  # Default to ego speed until history builds up
            
        # Compute scale expansion rate over the history window
        old_time, old_height = track_history[0]
        time_diff = timestamp - old_time
        
        if time_diff <= 0:
            return ego_speed_kmh
            
        # Rate of height change per second (pixels/sec)
        height_velocity = (box_height - old_height) / time_diff
        
        # Scale-rate heuristic: Faster expansion = higher relative closing speed
        expansion_factor = height_velocity / old_height
        
        # Estimate relative speed contribution (-factor because expansion means closing in)
        estimated_relative_speed = -expansion_factor * 120.0  
        estimated_speed = ego_speed_kmh + estimated_relative_speed
        
        # Clamp realistic vehicle speeds between 0 km/h and 160 km/h
        return round(max(0.0, min(160.0, estimated_speed)), 1)

    def cleanup_stale_tracks(self, active_track_ids):
        stale_ids = [tid for tid in self.history if tid not in active_track_ids]
        for tid in stale_ids:
            del self.history[tid]


class AIDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.4):
        print(f"[INFO] Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thresh = confidence_threshold
        self.speed_estimator = FrontVehicleSpeedEstimator()
        
        # Road user classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]

    def process_frame(self, frame, ego_speed_kmh=50.0):
        """
        Processes a video frame, tracks vehicles and pedestrians with ByteTrack, 
        and extracts spatial distance, collision trajectory, and kinematics.
        """
        timestamp = time.time()
        height, width, _ = frame.shape
        center_x_frame = width / 2.0
        
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=self.conf_thresh, 
            imgsz=480,
            verbose=False
        )
        
        detected_objects = []
        active_track_ids = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            classes = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            
            for box, cls_id, conf, track_id in zip(xyxy, classes, confs, track_ids):
                if cls_id in self.vehicle_classes and track_id is not None:
                    active_track_ids.append(track_id)
                    x1, y1, x2, y2 = box
                    box_width = max(1.0, x2 - x1)
                    box_height = max(1.0, y2 - y1)
                    obj_center_x = (x1 + x2) / 2.0
                    class_name = self.model.names[cls_id]

                    is_ped = (cls_id in [0, 1]) or (class_name.lower() in ["person", "bicycle"])
                    
                    if is_ped:
                        # Human distance estimation based on average human height (1.7m) & perspective
                        dist_m = max(1.0, round((height * 1.4) / box_height, 1))
                        # Direct collision corridor (narrow path right in front of vehicle, +-16% of width)
                        in_corridor = abs(obj_center_x - center_x_frame) < (width * 0.16)
                        is_centered = in_corridor
                        
                        # Only consider an imminent impact threat if in the driving path AND dangerously close
                        is_imminent_hit = in_corridor and (dist_m <= 4.5 or box_height >= (height * 0.28) or y2 >= (height * 0.70))
                        speed_kmh = 0.0
                    else:
                        # Vehicle distance estimation
                        dist_m = max(1.0, round((height * 1.6) / box_height, 1))
                        # Vehicle lane corridor
                        is_centered = abs(obj_center_x - center_x_frame) < (width * 0.22)
                        in_corridor = is_centered
                        is_imminent_hit = is_centered and (dist_m <= 4.0 or box_height >= (height * 0.48))
                        # Calculate vehicle speed
                        speed_kmh = self.speed_estimator.update_and_calculate(track_id, box, timestamp, ego_speed_kmh)
                    
                    obj_data = {
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 2),
                        "bbox": box,
                        "speed_kmh": speed_kmh,
                        "distance_m": dist_m,
                        "is_pedestrian": is_ped,
                        "is_front_vehicle": is_centered,
                        "in_collision_corridor": in_corridor,
                        "is_imminent_hit": is_imminent_hit,
                        "frame_height": height,
                        "frame_width": width
                    }
                    detected_objects.append(obj_data)
                    
        self.speed_estimator.cleanup_stale_tracks(active_track_ids)
        return detected_objects