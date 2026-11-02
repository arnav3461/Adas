import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database import get_all_incidents
except ImportError:
    def get_all_incidents(db_path="fleet_telematics.db"):
        return []

class CoachingSystem:
    def __init__(self, db_path="fleet_telematics.db"):
        self.db_path = db_path

    def get_driver_profile(self, incidents=None):
        """Analyzes logged incidents and generates risk profiling and targeted coaching recommendations."""
        if incidents is None:
            try:
                incidents = get_all_incidents(db_path=self.db_path)
            except Exception:
                incidents = []
            
        total_incidents = len(incidents)
        
        # Calculate dynamic risk score based on incident severities
        risk_points = 0
        counts = {
            "Collision_Risk": 0,
            "Harsh_Braking": 0,
            "Tailgating": 0,
            "Pedestrian_Hazard": 0,
            "Harsh_Acceleration": 0,
            "Other": 0
        }

        for inc in incidents:
            # inc: (id, timestamp, event_type, severity, description, video_clip_path, status)
            event_type = str(inc[2]) if len(inc) > 2 else "Unknown"
            severity = str(inc[3]).lower() if len(inc) > 3 else "medium"

            if "critical" in severity:
                risk_points += 25
            elif "high" in severity:
                risk_points += 15
            else:
                risk_points += 10

            matched = False
            for key in ["Collision_Risk", "Harsh_Braking", "Tailgating", "Pedestrian_Hazard", "Harsh_Acceleration"]:
                if key.lower() in event_type.lower():
                    counts[key] += 1
                    matched = True
                    break
            if not matched:
                counts["Other"] += 1

        risk_score = min(100, max(0, risk_points))
        safety_score = max(0, 100 - risk_score)

        if safety_score >= 90:
            grade = "A+ (Excellent)"
            status_label = "Safe Driver"
        elif safety_score >= 75:
            grade = "B (Good)"
            status_label = "Low Risk"
        elif safety_score >= 50:
            grade = "C (Fair)"
            status_label = "Attention Required"
        else:
            grade = "D (Hazardous)"
            status_label = "High Risk Alert"

        recommendations = []
        if total_incidents == 0:
            recommendations.append("🌟 **Flawless Driving Profile:** No incidents or safety anomalies detected. Maintain smooth headway and active road awareness.")
        else:
            if counts["Collision_Risk"] > 0:
                recommendations.append(f"⚠️ **Forward Collision Risk ({counts['Collision_Risk']} logged):** Rapid closing rates detected. Expand following distance to at least 3-4 seconds behind leading traffic.")
            if counts["Harsh_Braking"] > 0:
                recommendations.append(f"🛑 **Smooth Braking Advisory ({counts['Harsh_Braking']} logged):** Sudden deceleration spikes detected. Scan 10-15 seconds ahead to anticipate stops smoothly.")
            if counts["Tailgating"] > 0:
                recommendations.append(f"📏 **Buffer Zone Warning ({counts['Tailgating']} logged):** Operating inside hazardous vehicle proximity. Maintain a safe spatial cushion.")
            if counts["Pedestrian_Hazard"] > 0:
                recommendations.append(f"🚶 **Pedestrian Safety Alert ({counts['Pedestrian_Hazard']} logged):** Vulnerable road users detected in vehicle path. Exercise extreme vigilance in pedestrian zones.")
            if counts["Harsh_Acceleration"] > 0:
                recommendations.append(f"⚡ **Smooth Acceleration Guidance ({counts['Harsh_Acceleration']} logged):** Rapid throttle inputs detected. Apply gradual accelerator pressure to optimize fuel and safety.")
            recommendations.append("📊 **Telematics Feedback:** Consistent, gradual control inputs preserve vehicle stability and minimize accident probability.")

        return {
            "total_incidents": total_incidents,
            "risk_score": risk_score,
            "safety_score": safety_score,
            "grade": grade,
            "status_label": status_label,
            "counts": counts,
            "recommendations": recommendations,
            "incidents": incidents
        }