# modules/faceperson_detection.py

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

def run():
    # Load YOLOv8 (person detector)
    yolo_model = YOLO("yolov8n.pt")

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    # Animation setup
    animation_frames = []
    for i in range(15):
        frame = np.zeros((80, 450, 3), dtype=np.uint8)
        color = (0, 255 - i*15, i*15)
        cv2.putText(frame, "You may ask your doubts", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        animation_frames.append(frame)

    animation_index = 0
    show_animation = False
    animation_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        person_boxes = []
        hand_raised_detected = False

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if yolo_model.names[int(cls)] == "person":
                    x1, y1, x2, y2 = map(int, box)
                    person_boxes.append((x1, y1, x2, y2))

        for (x1, y1, x2, y2) in person_boxes:
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # Process hands
            img_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(person_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist_y = hand_landmarks.landmark[0].y
                    middle_tip_y = hand_landmarks.landmark[12].y

                    if middle_tip_y < wrist_y:
                        hand_raised_detected = True
                        cv2.putText(frame, "Hand Raised", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if hand_raised_detected:
            show_animation = True
            animation_counter = 30

        if show_animation:
            anim_frame = animation_frames[animation_index]
            animation_index = (animation_index + 1) % len(animation_frames)

            anim_resized = cv2.resize(anim_frame, (450, 80))
            h, w, _ = frame.shape
            y_offset = h - 90
            x_offset = 10

            frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

            animation_counter -= 1
            if animation_counter <= 0:
                show_animation = False

        cv2.imshow("Face + Person + Hand Raise Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
