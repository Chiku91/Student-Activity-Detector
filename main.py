# main.py

import streamlit as st
import threading
import time

# Import each module explicitly
from modules import (
    activity,
    hand_raise,
    speaking,
    standing,
    teacher_guidance,
    faceperson_detection,
    listening,
    head_down_writing,
    turning_head
)

# Map modes to functions
MODULE_FUNCTIONS = {
    "Writing + Object Detection": activity.run,
    "Hand Raise Detection": hand_raise.run,
    "Speaking Detection (Voice + Face)": speaking.run,
    "Standing Student Detection": standing.run,
    "Teacher Guiding Detection": teacher_guidance.run,
    "Face + Person + Hand Raise": faceperson_detection.run,
    "Listening + Concept Visuals": listening.run,
    "Head Down While Writing": head_down_writing.run,
    "Turning Head Detection": turning_head.run
}

# Function to run the selected module in a new thread
def run_selected_module(selected_module):
    st.success(f"Launching {selected_module}... close the webcam window to return.")
    MODULE_FUNCTIONS[selected_module]()

# =======================
# 🎨 Streamlit UI
# =======================

st.set_page_config(page_title="📹 Student Activity Detector", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4A5CFF;'>🎓 Student Activity Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar selection
selected_module = st.selectbox("📌 Select an Activity Module to Run", list(MODULE_FUNCTIONS.keys()))

if st.button("▶️ Run Module"):
    st.warning("Initializing webcam module... please wait.")
    # Run the module in a thread so the UI doesn't freeze
    threading.Thread(target=run_selected_module, args=(selected_module,), daemon=True).start()
    with st.spinner("Module running..."):
        time.sleep(2)
        st.info("Check the webcam window. Press 'Q' or 'ESC' in the webcam feed to close it.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, OpenCV, MediaPipe, YOLO, and PyTorch.")
