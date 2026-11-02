import time
import numpy as np

class AnomalyDetector:
    def __init__(self, ttc_threshold_seconds=2.5, cooldown_seconds=2.5):
        self.ttc_threshold = ttc_threshold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.tracked_history = {}  # track_id -> list of (timestamp, box_area, box_height)
        self.last_event_timestamps = {}  # event_type -> timestamp
        
        self.last_speed = None
        self.last_speed_time = None

    def evaluate(self, detections, sync_data, frame_shape=(360, 640)):
        """
        Unpacks sync data, evaluates vision and sensor metrics,
        and returns the highest-priority incident event if triggered.
        """
        if not sync_data:
            sync_data = {"timestamp": time.time(), "gps": {"speed": 40.0}}

        timestamp = sync_data.get("timestamp", time.time())
        ego_speed = sync_data.get("gps", {}).get("speed", 0.0)
        imu = sync_data.get("imu", {})

        events = []

        # 1. Evaluate Vision-Based Anomalies (Collision Risk, Tailgating, Pedestrian Hazard)
        vision_events = self._analyze_vision_tracks(detections, ego_speed, timestamp, frame_shape=frame_shape)
        events.extend(vision_events)

        # 2. Evaluate Telemetry Anomalies (Harsh Braking & Harsh Acceleration)
        telemetry_event = self._analyze_telemetry(ego_speed, imu, timestamp)
        if telemetry_event:
            events.append(telemetry_event)

        # Filter events through cooldown to prevent duplicate event spam
        valid_events = []
        for ev in events:
            ev_type = ev["event_type"]
            last_t = self.last_event_timestamps.get(ev_type, 0.0)
            if (timestamp - last_t) >= self.cooldown_seconds:
                self.last_event_timestamps[ev_type] = timestamp
                valid_events.append(ev)

        # Return highest severity event if any
        if not valid_events:
            return None

        # Sort: Critical > High > Medium
        severity_order = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        valid_events.sort(key=lambda x: severity_order.get(x.get("severity", "Medium"), 1), reverse=True)
        return valid_events[0]

    def _analyze_vision_tracks(self, tracks, ego_speed, timestamp, frame_shape=(360, 640)):
        events = []
        if not tracks:
            return events

        height, width = frame_shape[0], frame_shape[1]
        center_x_frame = width / 2.0

        # Clean old track history
        current_ids = set()
        for track in tracks:
            tid = track.get("track_id") if isinstance(track, dict) else (track[4] if len(track) >= 5 else 0)
            if tid:
                current_ids.add(tid)
        
        for tid in list(self.tracked_history.keys()):
            if tid not in current_ids:
                del self.tracked_history[tid]

        for track in tracks:
            try:
                if isinstance(track, dict):
                    bbox = track.get("bbox") or track.get("box") or [0, 0, 0, 0]
                    track_id = track.get("track_id", 0)
                    class_name = track.get("class_name", "vehicle").lower()
                    is_front = track.get("is_front_vehicle", False)
                    in_collision_corridor = track.get("in_collision_corridor", is_front)
                    dist_m = track.get("distance_m", None)
                    is_imminent_hit = track.get("is_imminent_hit", False)
                else:
                    bbox = track[:4]
                    track_id = track[4] if len(track) >= 5 else 0
                    class_name = "vehicle"
                    is_front = True
                    in_collision_corridor = True
                    dist_m = None
                    is_imminent_hit = False

                x1, y1, x2, y2 = bbox
                box_width = max(1.0, x2 - x1)
                box_height = max(1.0, y2 - y1)
                box_area = box_width * box_height
                obj_center_x = (x1 + x2) / 2.0

                is_ped = any(p in class_name for p in ["person", "pedestrian", "bicycle"])

                if dist_m is None:
                    scale = 1.4 if is_ped else 1.6
                    dist_m = round((height * scale) / box_height, 1)

                # =============================================================
                # PEDESTRIAN / CYCLIST HAZARD EVALUATION:
                # RULE: If a person is NOT too close, DO NOT detect as an incident.
                # ONLY detect if the person is going to get hit!
                # =============================================================
                if is_ped:
                    # 1. Must be directly in the vehicle's driving path corridor.
                    # Exclude pedestrians safely on the sidewalk or oncoming side.
                    in_corridor = in_collision_corridor or (abs(obj_center_x - center_x_frame) < (width * 0.16))
                    if not in_corridor:
                        # Person is safely off to the side -> NO INCIDENT
                        continue

                    # Maintain tracking history for relative velocity / TTC estimation
                    if track_id not in self.tracked_history:
                        self.tracked_history[track_id] = []
                    history = self.tracked_history[track_id]
                    history.append((timestamp, box_area, box_height))
                    self.tracked_history[track_id] = [h for h in history if timestamp - h[0] <= 1.5]

                    ttc = 999.0
                    dh_dt = 0.0
                    if len(history) >= 3:
                        t0, _, h0 = history[0]
                        dt = timestamp - t0
                        if dt > 0.08:
                            dh_dt = (box_height - h0) / dt
                            if dh_dt > 8.0:
                                ttc = box_height / dh_dt

                    # Condition A: Imminent Physical Impact (Person dangerously close in driving lane)
                    # Feet right in front of bumper (y2 > 0.70 of frame) or distance <= 4.5m or large bounding box
                    close_proximity_hit = (dist_m <= 4.5 or y2 >= (height * 0.70) or box_height >= (height * 0.28) or is_imminent_hit)

                    # Condition B: Active Closing Collision Course
                    # Vehicle moving towards pedestrian who is within 8 meters with imminent TTC (< 1.8s)
                    trajectory_hit = (ttc < 1.8 and dist_m <= 8.0 and (ego_speed > 8.0 or dh_dt > 12.0))

                    # ONLY detect if either Condition A or Condition B is satisfied!
                    if close_proximity_hit or trajectory_hit:
                        severity = "Critical" if (dist_m <= 3.5 or ttc < 1.0) else "High"
                        events.append({
                            "event_type": "Pedestrian_Hazard",
                            "severity": severity,
                            "timestamp": timestamp,
                            "description": f"IMMINENT PEDESTRIAN IMPACT: Vulnerable road user directly in driving path ({dist_m:.1f}m ahead, TTC: {min(ttc, 2.0):.1f}s)"
                        })
                    else:
                        # Person is at a safe distance or crossing outside collision path -> DO NOT DETECT
                        pass

                    # Finished checking pedestrian, skip vehicle logic
                    continue

                # =============================================================
                # VEHICLE HAZARD EVALUATION (Forward Collision Warning & Tailgating)
                # =============================================================
                if is_front:
                    if track_id not in self.tracked_history:
                        self.tracked_history[track_id] = []

                    history = self.tracked_history[track_id]
                    history.append((timestamp, box_area, box_height))
                    self.tracked_history[track_id] = [h for h in history if timestamp - h[0] <= 1.5]

                    if len(history) >= 4:
                        t0, _, h0 = history[0]
                        dt = timestamp - t0
                        if dt > 0.08:
                            dh_dt = (box_height - h0) / dt
                            if dh_dt > 15.0:  # Rapidly expanding bounding box
                                ttc = box_height / dh_dt
                                if ttc < self.ttc_threshold and ego_speed > 15.0:
                                    severity = "Critical" if ttc < 1.2 else "High"
                                    events.append({
                                        "event_type": "Collision_Risk",
                                        "severity": severity,
                                        "timestamp": timestamp,
                                        "description": f"Forward Collision Warning: TTC {ttc:.2f}s with lead vehicle (ID {track_id})"
                                    })

                    # Tailgating: Only trigger when driving at speed and encroaching on safety cushion
                    if (box_height > (height * 0.48) or dist_m <= 4.0) and ego_speed > 25.0:
                        events.append({
                            "event_type": "Tailgating",
                            "severity": "High",
                            "timestamp": timestamp,
                            "description": f"Hazardous following distance: Front vehicle ID {track_id} occupying forward safety cushion ({dist_m:.1f}m)"
                        })

            except Exception:
                continue

        return events

    def _analyze_telemetry(self, ego_speed, imu, timestamp):
        event = None

        if self.last_speed is not None and self.last_speed_time is not None:
            dt = timestamp - self.last_speed_time
            if 0.1 <= dt <= 2.0:
                accel_kmh_s = (ego_speed - self.last_speed) / dt

                # Harsh braking threshold: speed drops faster than 14 km/h per second
                if accel_kmh_s < -14.0:
                    event = {
                        "event_type": "Harsh_Braking",
                        "severity": "High",
                        "timestamp": timestamp,
                        "description": f"Harsh braking event detected: Deceleration rate {abs(accel_kmh_s):.1f} km/h/s"
                    }
                # Harsh acceleration threshold: speed jumps faster than 18 km/h per second
                elif accel_kmh_s > 18.0:
                    event = {
                        "event_type": "Harsh_Acceleration",
                        "severity": "Medium",
                        "timestamp": timestamp,
                        "description": f"Harsh acceleration detected: Throttle spike {accel_kmh_s:.1f} km/h/s"
                    }

        # Check IMU acceleration if available
        if not event and imu:
            accel = imu.get("accel", (0, 0, 9.8))
            if len(accel) >= 3 and accel[0] < -4.0:
                event = {
                    "event_type": "Harsh_Braking",
                    "severity": "High",
                    "timestamp": timestamp,
                    "description": f"IMU G-Force spike: Severe deceleration ({accel[0]:.2f} m/s²)"
                }

        self.last_speed = ego_speed
        self.last_speed_time = timestamp
        return event