# Aegis 🚛⚡👁️

### Enterprise Fleet Telematics & Real-Time Computer Vision Safety Platform

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Computer_Vision-YOLOv8-00FFFF?style=for-the-badge&logo=opencv&logoColor=white)
![PyDeck](https://img.shields.io/badge/3D_Mapping-PyDeck-1E1E1E?style=for-the-badge)
![OSRM](https://img.shields.io/badge/Routing-OSRM-1DB954?style=for-the-badge)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 📌 Overview

**Aegis** is an advanced fleet intelligence and real-time safety monitoring platform designed for commercial transportation and logistics operations.

By combining **YOLOv8 computer vision**, **IMU telemetry analysis**, **3D route visualization**, and **AI-powered driver behavior analytics**, Aegis transforms raw dashcam footage and vehicle sensor data into actionable safety insights, automated incident detection, and proactive driver coaching.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        AEGIS PLATFORM                       │
└─────────────────────────────────────────────────────────────┘

 High-Res Dashcam Video
            │
            ▼
      YOLOv8 Vision Engine
            │
            ▼
   Hazard & Proximity Detection
            │
            ├────────────────────────┐
            ▼                        ▼

     IMU Telemetry         Incident Detection Engine
 (Acceleration, G-Force)   (Safety Risk Analytics)

            └────────────────────────┘
                         │
                         ▼

┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
├─────────────────────────────────────────────────────────────┤
│  Tab 1 : Live Vision & Route Tracking                       │
│  Tab 2 : Incident Review & Database Management              │
│  Tab 3 : AI Coaching & Safety Score Analytics               │
└─────────────────────────────────────────────────────────────┘

                         │
                         ▼

               SQLite Data Persistence
```

---

## 🚀 Key Features

### 🎥 Real-Time Computer Vision

- YOLOv8-powered object detection
- Vehicle, truck, pedestrian, and obstacle tracking
- Collision-risk and proximity analysis
- Live video streaming within the dashboard

### 🛰️ Fleet Tracking & Navigation

- Real-time route visualization
- Uber-style trip progress tracking
- Green route = completed journey
- Gray route = remaining route
- OSRM route generation
- OpenStreetMap geocoding integration

### 🚚 3D Vehicle Visualization

- Interactive PyDeck-powered maps
- 3D truck and vehicle models
- Live vehicle position updates
- Origin, destination, and waypoint visualization

### ⚠️ Safety & Incident Management

- Hazard event detection
- Driver behavior monitoring
- Incident severity scoring
- Automated incident logging
- Review workflow for validation

### 💾 Database & Persistence

SQLite-powered storage for:

- Incident records
- GPS coordinates
- Vehicle telemetry
- Driver safety metrics
- Review status tracking

Supported statuses:

- `unreviewed`
- `true_positive`
- `false_positive`
- `edge_case`

---

## 📊 Dashboard Modules

### 1️⃣ Live Vision & Route Tracking

Provides:

- Real-time YOLOv8 detections
- Interactive route visualization
- Vehicle tracking
- GPS mapping
- Hazard localization

### 2️⃣ Incident Review Center

Operators can:

- Inspect detected anomalies
- Review incident severity
- Validate AI detections
- Update incident classifications
- Manage investigation workflows

### 3️⃣ AI Driver Coaching & Analytics

Includes:

- Driver safety score calculation
- Harsh braking detection
- Harsh acceleration analysis
- Safety trend monitoring
- Behavioral coaching recommendations

---

## 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Frontend/UI | Streamlit |
| Computer Vision | YOLOv8 |
| Mapping | PyDeck |
| Routing | OSRM |
| Geocoding | OpenStreetMap Nominatim |
| Database | SQLite |
| Telemetry | IMU Sensors |
| Analytics | Python Data Processing |

---

## 📁 Project Structure

```text
Aegis/
│
├── app.py
├── requirements.txt
├── fleet_telematics.db
│
├── assets/
├── models/
├── videos/
├── data/
├── database/
├── analytics/
└── documentation/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Aegis.git
cd Aegis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

---

## 📸 Core Capabilities

✅ Real-Time Object Detection

✅ Driver Safety Monitoring

✅ Fleet Telemetry Analysis

✅ Incident Management

✅ Route Tracking

✅ 3D Vehicle Visualization

✅ SQLite Data Persistence

✅ AI-Powered Behavioral Coaching

---

## 🎯 Use Cases

- Fleet Safety Operations
- ADAS Monitoring
- Driver Behavior Analytics
- Logistics Optimization
- Risk Assessment
- Incident Investigation
- Transportation Intelligence

---

## 🔮 Future Enhancements

- PostgreSQL Integration
- Cloud Deployment
- Multi-Vehicle Fleet Monitoring
- Predictive Risk Scoring
- AI Incident Summaries
- Edge Device Deployment
- Real-Time Notifications & Alerts

---

## 📄 License

This project is licensed under the MIT License.

---

## 🌍 Vision

**Aegis aims to build safer, smarter, and more proactive transportation systems through the fusion of AI, computer vision, telemetry, and intelligent analytics.**
