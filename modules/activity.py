# modules/activity.py

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

def run():
    # Initialize YOLOv8 model
    yolo = YOLO("yolov8n.pt")
    object_classes = {"laptop", "cell phone", "keyboard"}

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Animation for forbidden object usage
    animation_text = "Don't use laptop/phone in class, concentrate!"
    animation_frames = []
    for i in range(15):
        anim_frame = np.zeros((80, 600, 3), dtype=np.uint8)
        color = (0, 0, 255 - i * 15) if i < 8 else (0, 0, (i - 7) * 15)
        cv2.putText(anim_frame, animation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        animation_frames.append(anim_frame)
    animation_index = 0

    def hand_box_from_landmarks(landmarks, w, h):
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        results = list(yolo.track(source=frame, persist=True, stream=False, verbose=False))

        forbidden_object_detected = False
        person_boxes = []

        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            ids_tensor = r.boxes.id
            ids = ids_tensor.tolist() if ids_tensor is not None else [None] * len(r.boxes.xyxy)
            for box, cls, tid in zip(r.boxes.xyxy, r.boxes.cls, ids):
                label = yolo.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                if label in object_classes:
                    forbidden_object_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
                if label == "person" and tid is not None:
                    person_boxes.append((int(tid), x1, y1, x2, y2))

        # Show warning animation
        if forbidden_object_detected:
            anim_frame = animation_frames[animation_index]
            animation_index = (animation_index + 1) % len(animation_frames)
            anim_resized = cv2.resize(anim_frame, (min(frame_w - 20, 600), 80))
            y_offset = frame_h - 90
            x_offset = (frame_w - anim_resized.shape[1]) // 2
            frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

        # Writing detection
        for pid, x1, y1, x2, y2 in person_boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            eye_level_global = y1 + int(0.25 * (y2 - y1))
            cv2.line(frame, (x1, eye_level_global), (x2, eye_level_global), (200, 200, 200), 2)
            hr = hands.process(roi_rgb)
            writing = False

            if hr.multi_hand_landmarks:
                for hl in hr.multi_hand_landmarks:
                    mp_draw.draw_landmarks(roi, hl, mp_hands.HAND_CONNECTIONS)
                    _, _, _, hy2 = hand_box_from_landmarks(hl.landmark, x2 - x1, y2 - y1)
                    global_hand_bottom = hy2 + y1
                    if global_hand_bottom > eye_level_global:
                        writing = True
                        break

            color = (0, 255, 0) if writing else (0, 0, 255)
            label_text = f"ID {pid}: {'Writing' if writing else 'Not Writing'}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Writing & Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
