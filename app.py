import sys
import os

# Force Python to look in the current script directory for local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import time
import numpy as np
from datetime import datetime

# Import project modules
from sensor_pipeline import SensorDataPipeline
from ai_detector import AIDetector
from Anomalydetect import AnomalyDetector
from EvidenceGenerator import EvidenceGenerator
from database import Database, get_all_incidents, insert_incident, update_incident_status, delete_incident, clear_all_incidents
from coaching import CoachingSystem

# Optional import for learning pipeline
try:
    from learning_pipeline import ContinuousLearningPipeline
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Aegis - Fleet Telematics & ADAS Dashboard",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Aegis Fleet Telematics & ADAS Dashboard")
st.caption("⚡ Real-time Perception • Smooth 30+ FPS Tracking • Live Telematics • AI Driver Coaching")

# --- Initialize session state ---
if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = []
if "current_trip_count" not in st.session_state:
    st.session_state.current_trip_count = 0
if "active_hazard" not in st.session_state:
    st.session_state.active_hazard = None

# --- Helper Rendering Functions for Tab 2 and Tab 3 ---
def render_incident_review_ui(incidents=None, key_prefix="rev"):
    """Renders the Incident Review tab contents once per page execution."""
    if incidents is None:
        incidents = get_all_incidents()

    st.subheader("📋 Trip Incident Review Center")
    st.write("Inspect, validate, and manage safety anomalies detected during live trips.")

    total = len(incidents)
    unreviewed = sum(1 for i in incidents if len(i) > 6 and i[6] == "unreviewed")
    true_pos = sum(1 for i in incidents if len(i) > 6 and i[6] == "true_positive")
    false_pos = sum(1 for i in incidents if len(i) > 6 and i[6] == "false_positive")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Incidents", total)
    c2.metric("Pending Review", unreviewed)
    c3.metric("True Positives", true_pos)
    c4.metric("False Positives", false_pos)

    st.divider()

    if incidents:
        for idx, inc in enumerate(incidents):
            inc_id, ts, event_type, severity, desc, clip_path, status = inc
            time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            
            # Severity badge colors
            sev_color = "#ff4d4d" if "critical" in str(severity).lower() else ("#ffa64d" if "high" in str(severity).lower() else "#ffcc00")
            status_color = "#00e676" if status == "true_positive" else ("#ff5252" if status == "false_positive" else "#90caf9")

            expander_label = f"[{time_str}] {event_type}  |  Severity: {severity}  |  Status: {status.upper()}"
            with st.expander(expander_label, expanded=(idx == 0)):
                col_det, col_media = st.columns([1, 1])

                with col_det:
                    st.markdown(f"**Description:** {desc}")
                    st.markdown(f"**Trigger Time:** `{time_str}` (UTC: `{ts:.2f}`)")
                    st.markdown(f"**Severity Level:** <span style='color:{sev_color}; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Review Classification:** <span style='color:{status_color}; font-weight:bold;'>{status.upper()}</span>", unsafe_allow_html=True)

                    btn_c1, btn_c2, btn_c3 = st.columns(3)
                    if btn_c1.button("✅ True Pos", key=f"{key_prefix}_tp_{inc_id}_{idx}"):
                        update_incident_status(inc_id, "true_positive")
                        st.rerun()
                    if btn_c2.button("❌ False Pos", key=f"{key_prefix}_fp_{inc_id}_{idx}"):
                        update_incident_status(inc_id, "false_positive")
                        st.rerun()
                    if btn_c3.button("🗑️ Delete", key=f"{key_prefix}_del_{inc_id}_{idx}"):
                        delete_incident(inc_id)
                        st.rerun()

                with col_media:
                    media_t1, media_t2 = st.tabs(["📸 Snapshot", "🎬 Video Clip"])
                    image_snapshot_path = clip_path.replace(".mp4", ".jpg") if clip_path else ""
                    with media_t1:
                        if image_snapshot_path and os.path.exists(image_snapshot_path):
                            st.image(image_snapshot_path, caption=f"Trigger Snapshot: {event_type}", use_container_width=True)
                        else:
                            st.info("ℹ️ Telemetry logged in database.")
                    with media_t2:
                        if clip_path and os.path.exists(clip_path):
                            st.video(clip_path)
                        else:
                            st.info("ℹ️ Video clip not available or encoding in background.")

        st.write("")
        c_clear, c_ref = st.columns([1, 1])
        if c_clear.button("🧹 Clear All Incidents", key=f"{key_prefix}_clear_all_btn"):
            clear_all_incidents()
            st.rerun()
        if c_ref.button("🔄 Refresh List", key=f"{key_prefix}_refresh_btn"):
            st.rerun()
    else:
        st.info("💡 **No incidents recorded yet.** Start the live trip pipeline or trigger a hazard simulation to log real-time events.")


def render_driver_coaching_ui(container, incidents=None):
    """Dynamically renders the Driver Coaching tab in real time."""
    if incidents is None:
        incidents = get_all_incidents()

    coaching_sys = CoachingSystem()
    profile = coaching_sys.get_driver_profile(incidents=incidents)

    with container.container():
        st.subheader("🎯 Post-Trip & Live Driver Coaching")
        st.write("Behavioral risk profiling, driver safety score, and actionable AI coaching suggestions.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trip Incidents", profile["total_incidents"])
        
        # Color score
        risk = profile["risk_score"]
        safety = profile["safety_score"]
        col2.metric("Safety Score", f"{safety} / 100", delta=f"{profile['status_label']}", delta_color="normal" if safety >= 70 else "inverse")
        col3.metric("Driver Risk Score", f"{risk} / 100", delta="High Risk" if risk > 40 else "Acceptable", delta_color="inverse" if risk > 40 else "normal")
        col4.metric("Safety Grade", profile["grade"])

        # Visual Safety Score Bar
        st.markdown(f"""
            <div style="background-color: #222; border-radius: 8px; overflow: hidden; margin: 10px 0 20px 0; border: 1px solid #444;">
                <div style="width: {safety}%; background: linear-gradient(90deg, #ff4444 0%, #ffbb33 50%, #00C851 100%); height: 16px;"></div>
            </div>
        """, unsafe_allow_html=True)

        # Incident Breakdown
        st.markdown("### 📊 Safety Incident Breakdown")
        counts = profile["counts"]
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("💥 Collision Risk", counts.get("Collision_Risk", 0))
        b2.metric("🛑 Harsh Braking", counts.get("Harsh_Braking", 0))
        b3.metric("📏 Tailgating", counts.get("Tailgating", 0))
        b4.metric("🚶 Pedestrian Hazards", counts.get("Pedestrian_Hazard", 0))
        b5.metric("⚡ Harsh Accel", counts.get("Harsh_Acceleration", 0))

        # AI Coaching Recommendations
        st.markdown("### 💡 AI Driver Coaching Recommendations")
        for rec in profile["recommendations"]:
            st.info(rec)


# --- Navigation Tabs ---
tab_live, tab_review, tab_coaching = st.tabs([
    "🚗 Live Trip & ADAS Monitoring", 
    "📋 Incident Review & Dataset", 
    "🎯 Driver Coaching & Profiling"
])

# Tab 2: Rendered once per script execution (no duplicate keys!)
with tab_review:
    render_incident_review_ui()

# Tab 3: Rendered with reactive placeholder for real-time coaching updates
with tab_coaching:
    coaching_placeholder = st.empty()
    render_driver_coaching_ui(coaching_placeholder)

# ==========================================
# TAB 1: LIVE TRIP & ADAS MONITORING
# ==========================================
with tab_live:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed & Perception")
        video_placeholder = st.empty()
        
        st.markdown("### 🛞 Live Trip Speedometer & Active Hazard Alert")
        hazard_alert_placeholder = st.empty()
        speed_metric_placeholder = st.empty()
        
    with col2:
        st.subheader("📁 Video Dataset & Controls")
        
        main_video_file = "12604097_3840_2160_30fps.mp4"
        temp_video_path = "temp_uploaded_dataset.mp4"
        
        # Build source options: 1-click access to 12604097_3840_2160_30fps.mp4!
        source_options = []
        if os.path.exists(main_video_file):
            source_options.append(f"🚗 4K Dashcam Dataset ({main_video_file})")
        elif os.path.exists(temp_video_path):
            source_options.append("🚗 Active Driving Dataset (temp_uploaded_dataset.mp4)")
        source_options.append("📤 Upload Custom Driving Video")
        source_options.append("🎥 Live Camera (Webcam)")
        
        source_choice = st.radio("Select Video Input Source", source_options, index=0)
        
        active_video_source = None
        uploaded_file = None
        
        if main_video_file in source_choice:
            active_video_source = main_video_file
        elif "Active Driving Dataset" in source_choice:
            active_video_source = temp_video_path
        elif "Upload Custom" in source_choice:
            uploaded_file = st.file_uploader("Upload driving video (.mp4, .avi, .mov, .mkv)", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_file is not None:
                if st.session_state.get("current_uploaded_filename") != uploaded_file.name:
                    if "sensor_pipeline" in st.session_state:
                        st.session_state.sensor_pipeline.stop()
                        st.session_state.pop("sensor_pipeline", None)
                        st.session_state.pop("evidence_gen", None)
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state["current_uploaded_filename"] = uploaded_file.name
                    st.toast(f"✅ Loaded: {uploaded_file.name}", icon="📁")
                active_video_source = temp_video_path
        else:
            active_video_source = 0
            
        dataset_meta = None
        if active_video_source and isinstance(active_video_source, str) and os.path.exists(active_video_source):
            cap_check = cv2.VideoCapture(active_video_source)
            w_orig = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_orig = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_orig = round(cap_check.get(cv2.CAP_PROP_FPS) or 30.0, 1)
            total_f = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            dur_s = round(total_f / fps_orig, 1) if fps_orig > 0 else 0
            ret_p, prev_f = cap_check.read()
            cap_check.release()
            
            source_display_name = uploaded_file.name if uploaded_file else os.path.basename(active_video_source)
            dataset_meta = {
                "w": w_orig, "h": h_orig, "fps": fps_orig, 
                "frames": total_f, "duration": dur_s,
                "preview": prev_f,
                "source_name": source_display_name
            }
            res_tag = "4K UHD" if w_orig >= 3840 else ("1080p FHD" if w_orig >= 1920 else "HD")
            st.success(f"🎥 **Active Video:** `{source_display_name}`")
            mc1, mc2, mc3 = st.columns(3)
            mc1.caption(f"📐 **Input:** {w_orig}x{h_orig} ({res_tag})")
            mc2.caption(f"⚡ **Stream:** 1280x720 HD @ {fps_orig} FPS")
            mc3.caption(f"⏱️ **Duration:** {dur_s}s ({total_f} frames)")
        elif active_video_source == 0:
            st.info("🎥 Live Camera / Webcam selected as primary video stream.")
        
        # Pipeline Control Toggle
        run_pipeline = st.toggle("🚀 Start Live Trip Pipeline", value=False)

        st.subheader("⚡ Test Hazard Simulation")
        st.caption("Click to simulate hazards and verify instant real-time linking across tabs:")
        sim_col1, sim_col2 = st.columns(2)
        sim_collision = sim_col1.button("💥 Collision Risk", key="sim_btn_collision", help="Simulate a forward collision warning")
        sim_braking = sim_col2.button("🛑 Harsh Braking", key="sim_btn_braking", help="Simulate a harsh deceleration spike")
        sim_tailgating = sim_col1.button("📏 Tailgating", key="sim_btn_tailgating", help="Simulate dangerous following proximity")
        sim_pedestrian = sim_col2.button("🚶 Pedestrian (Imminent Hit)", key="sim_btn_pedestrian", help="Simulate pedestrian in immediate collision path (< 4m)")

        st.subheader("📊 Live Trip Safety Pulse")
        live_stats_placeholder = st.empty()

        st.subheader("🤖 AI Commentary Feed")
        ai_commentary_placeholder = st.empty()
        
        st.subheader("🛰️ Live Telematics")
        telemetry_placeholder = st.empty()

    # Handle manual simulation clicks
    if sim_collision or sim_braking or sim_tailgating or sim_pedestrian:
        sim_ts = time.time()
        time_str = datetime.fromtimestamp(sim_ts).strftime("%H:%M:%S")
        if sim_collision:
            ev_type, sev, desc = "Collision_Risk", "Critical", "Forward Collision Warning: TTC 1.1s with vehicle in path"
        elif sim_braking:
            ev_type, sev, desc = "Harsh_Braking", "High", "Severe deceleration: -16.4 km/h/s sudden brake apply"
        elif sim_tailgating:
            ev_type, sev, desc = "Tailgating", "High", "Unsafe following distance: operating inside safety buffer"
        else:
            ev_type, sev, desc = "Pedestrian_Hazard", "Critical", "IMMINENT PEDESTRIAN IMPACT: Vulnerable road user directly in collision trajectory (< 3.5m)"

        # Create dummy frame snapshot for evidence
        dummy_snapshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(dummy_snapshot, f"HAZARD: {ev_type}", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        cv2.putText(dummy_snapshot, f"Time: {time_str} | Severity: {sev}", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        os.makedirs("events_output", exist_ok=True)
        base_name = f"{ev_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        video_p = os.path.join("events_output", f"{base_name}.mp4")
        image_p = os.path.join("events_output", f"{base_name}.jpg")
        cv2.imwrite(image_p, dummy_snapshot)

        insert_incident(sim_ts, ev_type, sev, desc, video_p)
        st.session_state.incident_logs.insert(0, f"⚠️ **[{time_str}] SIMULATED INCIDENT ({ev_type})**: {desc}")
        st.toast(f"Logged Incident: {ev_type}", icon="🚨")
        
        # Clean rerun so Tab 1, Tab 2, and Tab 3 refresh with zero duplicate keys!
        st.rerun()

    # --- Live Trip Pipeline Execution ---
    if run_pipeline:
        target_source = active_video_source if active_video_source is not None else 0
        
        # If pipeline exists but source changed, restart it cleanly
        if "sensor_pipeline" in st.session_state:
            if getattr(st.session_state.sensor_pipeline, "video_source", None) != target_source:
                st.session_state.sensor_pipeline.stop()
                st.session_state.pop("sensor_pipeline", None)
                st.session_state.pop("evidence_gen", None)

        if "sensor_pipeline" not in st.session_state:
            st.session_state.sensor_pipeline = SensorDataPipeline(
                video_source=target_source, 
                frame_width=1280, 
                frame_height=720
            )
            st.session_state.ai_detector = AIDetector()
            st.session_state.anomaly_detector = AnomalyDetector()
            st.session_state.db = Database()
            st.session_state.sensor_pipeline.start()
            
        pipeline = st.session_state.sensor_pipeline
        
        if "evidence_gen" not in st.session_state:
            st.session_state.evidence_gen = EvidenceGenerator(pipeline=pipeline)

        ai_det = st.session_state.ai_detector
        anomaly_det = st.session_state.anomaly_detector
        evidence_gen = st.session_state.evidence_gen
        
        frame_counter = 0
        last_frame_id = -1
        cached_detections = []

        # Smooth Live Execution Stream Loop with monotonic frame_id tracking
        while pipeline.running:
            latest = pipeline.video_buffer.get_latest()
            if latest is None:
                time.sleep(0.003)
                continue
                
            frame_id, timestamp, frame = latest
            if frame_id == last_frame_id:
                time.sleep(0.003)
                continue
            last_frame_id = frame_id
            frame_counter += 1
            
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            
            sync_data = pipeline.get_synchronized_sample(timestamp)
            ego_speed = sync_data.get("gps", {}).get("speed", 0.0) if sync_data else 50.0
            
            # 🚀 HIGH-PERFORMANCE INFERENCE: Run YOLO tracking every 2nd frame for 30+ FPS smooth rendering
            if frame_counter % 2 == 0 or not cached_detections:
                detections = ai_det.process_frame(frame, ego_speed_kmh=ego_speed)
                cached_detections = detections
            else:
                detections = cached_detections
            
            display_speed = ego_speed
            front_vehicle_detected = False
            for obj in detections:
                if obj.get("is_front_vehicle", False):
                    display_speed = obj.get("speed_kmh", ego_speed)
                    front_vehicle_detected = True
                    break
            
            # Evaluate for multi-modal ADAS incidents
            event = anomaly_det.evaluate(detections, sync_data, frame_shape=frame.shape)
            
            if event:
                event_type = event['event_type']
                desc = event['description']
                severity = event.get('severity', 'High')
                
                ai_comment = f"⚠️ **[{time_str}] INCIDENT ({event_type})**: {desc} (Severity: {severity})"
                st.session_state.incident_logs.insert(0, ai_comment)
                st.session_state.active_hazard = (event_type, severity, desc, time.time())
                st.session_state.current_trip_count += 1
                
                # Annotate trigger frame with red hazard warning banner
                hazard_frame = frame.copy()
                cv2.putText(hazard_frame, f"INCIDENT: {event_type} ({severity})", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 🚀 ASYNCHRONOUS EVIDENCE GENERATION: Doesn't stall or stutter live video!
                try:
                    evidence_gen.generate_incident_evidence(hazard_frame, sync_data, event)
                except Exception as ex:
                    print(f"Evidence notice: {ex}")
                
                # Update Driver Coaching Tab in real time (coaching contains zero buttons, perfectly safe!)
                updated_incidents = get_all_incidents()
                render_driver_coaching_ui(coaching_placeholder, updated_incidents)
                
            elif front_vehicle_detected and display_speed < (ego_speed - 15):
                ai_comment = f"ℹ️ **[{time_str}] AI Advisory**: Closing gap with lead vehicle ({display_speed:.1f} vs {ego_speed:.1f} km/h)."
                if not st.session_state.incident_logs or "AI Advisory" not in st.session_state.incident_logs[0]:
                    st.session_state.incident_logs.insert(0, ai_comment)
            
            if len(st.session_state.incident_logs) > 10:
                st.session_state.incident_logs.pop()

            # Active Hazard Alert Banner
            if st.session_state.active_hazard and (time.time() - st.session_state.active_hazard[3]) < 3.5:
                hz_type, hz_sev, hz_desc, _ = st.session_state.active_hazard
                hazard_alert_placeholder.markdown(f"""
                    <div style="background-color: #8b0000; padding: 10px; border-radius: 6px; text-align: center; border: 2px solid #ff4444; margin-bottom: 8px;">
                        <span style="color: #ffffff; font-weight: bold; font-size: 15px;">🚨 ACTIVE HAZARD: {hz_type} ({hz_sev})</span><br/>
                        <span style="color: #ffcccc; font-size: 12px;">{hz_desc}</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                hazard_alert_placeholder.markdown("""
                    <div style="background-color: #004d26; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid #00c851; margin-bottom: 8px;">
                        <span style="color: #a3ffc2; font-weight: bold; font-size: 13px;">🟢 ALL SYSTEMS CLEAR • ADAS MONITORING ACTIVE</span>
                    </div>
                """, unsafe_allow_html=True)
                
            # Render AI Commentary Feed
            logs_html = "<div style='background-color: #111111; padding: 12px; border-radius: 8px; height: 160px; overflow-y: auto; border: 1px solid #333333;'>"
            if st.session_state.incident_logs:
                for log in st.session_state.incident_logs:
                    logs_html += f"<p style='font-size: 12px; color: #dddddd; margin-bottom: 6px;'>{log}</p>"
            else:
                logs_html += "<p style='color: #888888; font-style: italic; font-size: 12px;'>AI monitoring active... No incidents reported yet.</p>"
            logs_html += "</div>"
            ai_commentary_placeholder.markdown(logs_html, unsafe_allow_html=True)
                
            # Annotate video frame with bounding boxes and tracks
            annotated_frame = frame.copy()
            for obj in detections:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                is_front = obj.get("is_front_vehicle", False)
                cls_name = obj.get("class_name", "object")
                dist_m = obj.get("distance_m", 0.0)
                is_imminent = obj.get("is_imminent_hit", False)
                
                is_ped = "person" in cls_name.lower() or "bicycle" in cls_name.lower()
                if is_ped:
                    if is_imminent:
                        color = (0, 0, 255)       # Red: Imminent collision threat ("going to get hit")
                        thickness = 3
                        label = f"⚠️ IMPACT RISK: {cls_name} | {dist_m:.1f}m"
                    elif is_front:
                        color = (0, 165, 255)     # Orange: Ahead in corridor, but safe distance
                        thickness = 2
                        label = f"{cls_name} | {dist_m:.1f}m (Corridor)"
                    else:
                        color = (0, 255, 0)       # Green: Safe distance / sidewalk
                        thickness = 2
                        label = f"{cls_name} | {dist_m:.1f}m (Safe)"
                else:
                    if is_front and obj.get("speed_kmh", 0) < (ego_speed - 15):
                        color = (0, 0, 255)
                        thickness = 2
                        label = f"{cls_name} | {obj.get('speed_kmh', 0)} km/h ({dist_m:.1f}m) ⚠️"
                    else:
                        color = (0, 255, 0)
                        thickness = 2
                        label = f"{cls_name} | {obj.get('speed_kmh', 0)} km/h ({dist_m:.1f}m)"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, max(22, y1 - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            
            # High-performance JPEG byte streaming prevents browser websocket congestion
            _, encoded_frame = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            video_placeholder.image(encoded_frame.tobytes(), use_container_width=True)
            
            # Speedometer
            speed_metric_placeholder.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 14px; border-radius: 8px; text-align: center; border: 2px solid #00ffcc; box-shadow: 0px 4px 10px rgba(0,255,204,0.15);">
                    <p style="color: #aaaaaa; margin: 0; font-size: 11px; letter-spacing: 1px;">CURRENT TRIP SPEED</p>
                    <p style="color: #00ffcc; font-size: 38px; font-weight: bold; margin: 2px 0 0 0;">{display_speed:.1f} <span style="font-size: 16px; color: #ffffff;">km/h</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Live Trip Safety Pulse on Tab 1
            live_stats_placeholder.markdown(f"""
                <div style="background-color: #1a1a1a; padding: 10px; border-radius: 6px; border: 1px solid #333;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                        <span>Trip Incidents Logged:</span>
                        <strong style="color: #ff5252;">{st.session_state.current_trip_count}</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            gps_lat = sync_data.get("gps", {}).get("lat", 0.0) if sync_data else 0.0
            gps_lon = sync_data.get("gps", {}).get("lon", 0.0) if sync_data else 0.0
            telemetry_placeholder.markdown(f"""
                * **GPS Latitude:** `{gps_lat:.5f}`
                * **GPS Longitude:** `{gps_lon:.5f}`
                * **Ego Speed:** `{ego_speed:.1f} km/h`
                * **Active Road Tracks:** `{len(detections)}`
            """)

            time.sleep(0.002)
            
    else:
        if "sensor_pipeline" in st.session_state:
            st.session_state.sensor_pipeline.stop()
            st.session_state.pop("sensor_pipeline", None)
            st.session_state.pop("evidence_gen", None)
            st.session_state["trip_just_ended"] = True
            st.rerun()

        if st.session_state.pop("trip_just_ended", False):
            st.success("🏁 **Trip Ended Successfully!** All recorded incidents and driver coaching profile are fully synced in Tabs 2 and 3.")

        if dataset_meta and dataset_meta.get("preview") is not None:
            p_frame = dataset_meta["preview"]
            if p_frame.shape[1] != 1280 or p_frame.shape[0] != 720:
                p_frame = cv2.resize(p_frame, (1280, 720), interpolation=cv2.INTER_AREA)
            _, p_buf = cv2.imencode('.jpg', p_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            src_title = dataset_meta.get('source_name', 'Driving Video')
            video_placeholder.image(
                p_buf.tobytes(), 
                caption=f"🎥 Ready: {src_title} (720p HD Preview • Toggle 'Start Live Trip Pipeline' to Run)", 
                use_container_width=True
            )
        else:
            video_placeholder.info("Select a video source above (or use webcam) and toggle **'🚀 Start Live Trip Pipeline'** to begin analysis.")
        
        hazard_alert_placeholder.markdown("""
            <div style="background-color: #222222; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid #444444; margin-bottom: 8px;">
                <span style="color: #888888; font-size: 13px;">⏸️ SYSTEM PAUSED • AWAITING TRIP START</span>
            </div>
        """, unsafe_allow_html=True)

        speed_metric_placeholder.markdown("""
            <div style="background-color: #1e1e1e; padding: 14px; border-radius: 8px; text-align: center; border: 2px solid #555555;">
                <p style="color: #aaaaaa; margin: 0; font-size: 11px; letter-spacing: 1px;">CURRENT TRIP SPEED</p>
                <p style="color: #888888; font-size: 38px; font-weight: bold; margin: 2px 0 0 0;">0.0 <span style="font-size: 16px; color: #ffffff;">km/h</span></p>
            </div>
        """, unsafe_allow_html=True)
        
        ai_commentary_placeholder.markdown("""
            <div style='background-color: #111111; padding: 12px; border-radius: 8px; height: 160px; border: 1px solid #333333;'>
                <p style='color: #888888; font-style: italic; font-size: 12px;'>Pipeline paused. AI commentary will stream here during active trips.</p>
            </div>
        """, unsafe_allow_html=True)
        
        live_stats_placeholder.markdown(f"""
            <div style="background-color: #1a1a1a; padding: 10px; border-radius: 6px; border: 1px solid #333;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span>Database Total Incidents:</span>
                    <strong>{len(get_all_incidents())}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

        telemetry_placeholder.markdown("*System paused. Ready for live sensor ingestion.*")