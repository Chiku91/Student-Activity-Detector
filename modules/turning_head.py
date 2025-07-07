# modules/turning_head.py

import cv2
import mediapipe as mp
import time
import numpy as np

def run():
    # Initialize Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(0)

    # Turning logic variables
    head_turned_start_time = None
    HEAD_TURN_DURATION_THRESHOLD = 5  # seconds
    is_head_turned = False
    frame_counter = 0  # for blinking effect

    def is_head_turning(nose_x, left_cheek_x, right_cheek_x, threshold=0.15):
        left_dist = abs(nose_x - left_cheek_x)
        right_dist = abs(right_cheek_x - nose_x)
        if left_dist == 0 or right_dist == 0:
            return False
        ratio = left_dist / right_dist
        return ratio < (1 - threshold) or ratio > (1 + threshold)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        status = "Looking straight"
        show_warning = False
        current_time = time.time()

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                nose_x = face_landmarks.landmark[1].x
                left_cheek_x = face_landmarks.landmark[234].x
                right_cheek_x = face_landmarks.landmark[454].x

                nose_px = int(nose_x * w)
                left_px = int(left_cheek_x * w)
                right_px = int(right_cheek_x * w)

                cv2.circle(frame, (nose_px, int(h / 2)), 5, (255, 0, 0), -1)
                cv2.circle(frame, (left_px, int(h / 2)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (right_px, int(h / 2)), 5, (0, 0, 255), -1)

                turning = is_head_turning(nose_x, left_cheek_x, right_cheek_x)

                if turning:
                    if not is_head_turned:
                        head_turned_start_time = current_time
                        is_head_turned = True
                    elif current_time - head_turned_start_time >= HEAD_TURN_DURATION_THRESHOLD:
                        status = "Turned head"
                        show_warning = True
                else:
                    is_head_turned = False
                    head_turned_start_time = None
                    status = "Looking straight"

        # Status display
        color = (0, 255, 0) if status == "Looking straight" else (0, 0, 255)
        cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Blinking warning
        if show_warning and (frame_counter // 15) % 2 == 0:
            cv2.putText(frame, "Don't turn your head backward; look forward", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        frame_counter += 1
        cv2.imshow("Head Turn Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
