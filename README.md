This project is a real-time computer vision-based system designed to **monitor and analyze student activities in the classroom** using

The system detects and visualizes multiple student behaviors such as **writing**, **raising hands**, **speaking**, **standing**, **head turning**, and even **teacher guidance**, with animated feedback messages to enhance interactivity.

**🎯 This system is used for student activity detection, tracks student disengagement or confusion using facial expression analysis, and suggests interventions using dynamic prompts and warnings.**

## 🔧 Tech Stack

- [✔️] **OpenCV** – Video capture, processing, and visual overlays
- [✔️] **YOLOv5 / YOLOv8** – Object detection (person, laptop, etc.)
- [✔️] **MediaPipe** – Hands, Face Mesh, Pose tracking
- [✔️] **PyTorch** – Deep learning model inference
- [✔️] **SoundDevice** – Audio volume detection
- [✔️] **Streamlit** – Web interface and module launcher

## 🚀 How It Works/ Module Overview

| Module | Description |
|--------|-------------|
| `activity.py` | 📚 Detects **writing** posture and forbidden objects (phone/laptop). Shows red warning animation. |
| `hand_raise.py` | ✋ Detects **raised hands** within person ROIs using MediaPipe Hands. |
| `speaking.py` | 🗣️ Detects **students speaking** using a combination of **audio volume** and **face location**. |
| `standing.py` | 🚶 Detects if students are **standing** (based on bounding box aspect ratio). |
| `teacher_guidance.py` | 🧑‍🏫 Detects **teacher guiding students** by checking for **standing posture, hand raise, and open mouth**. |
| `faceperson_detection.py` | 🧍 Combines **person detection** with **hand-raise detection** and shows encouraging animation. |
| `listening.py` | 👂 Tracks **student attention** using **face presence + movement tracking**. Also rotates visual aids like **charts, graphs, equations**. |
| `head_down_writing.py` | 📓 Detects students **writing with head down** posture. Shows a warning to sit upright. |
| `turning_head.py` | 👀 Detects if students **turn their head sideways/backward** for more than 5 seconds. |
| `main.py` | 🧠 Streamlit-based GUI to launch all the modules easily from a dropdown menu. |

## 🖥️ Streamlit Interface

Run the app via:

bash
streamlit run main.py


