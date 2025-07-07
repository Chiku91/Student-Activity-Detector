# modules/teacher_guidance.py

import cv2
import mediapipe as mp
import torch
import numpy as np

def run():
    # Load YOLOv5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    # MediaPipe initializations
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    STANDING_HEIGHT_THRESHOLD = 150
    SITTING_HEIGHT_THRESHOLD = 100

    def is_hand_raised(landmarks):
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        return (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y)

    def is_mouth_open(face_landmarks, w, h, threshold=5):
        top_lip = face_landmarks.landmark[13]
        bottom_lip = face_landmarks.landmark[14]
        dist = np.linalg.norm(
            np.array([top_lip.x * w, top_lip.y * h]) -
            np.array([bottom_lip.x * w, bottom_lip.y * h])
        )
        return dist > threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb_frame)
        persons = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                height = y2 - y1
                persons.append({'box': (x1, y1, x2, y2), 'height': height})

        persons = sorted(persons, key=lambda p: p['height'], reverse=True)[:3]
        teacher_guiding_detected = False

        for person in persons:
            x1, y1, x2, y2 = person['box']
            height = person['height']
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            if height > STANDING_HEIGHT_THRESHOLD:
                # Possibly teacher standing
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(person_rgb)

                hand_raised = False
                if pose_results.pose_landmarks:
                    hand_raised = is_hand_raised(pose_results.pose_landmarks.landmark)

                face_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(face_rgb)

                mouth_open = False
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    mouth_open = is_mouth_open(face_landmarks, x2 - x1, y2 - y1)

                if hand_raised and mouth_open:
                    teacher_guiding_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.putText(frame, "Teacher Guiding", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

                    # Friendly message
                    msg = "Clear all your doubts, you can ask anything!"
                    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    break

            elif height < SITTING_HEIGHT_THRESHOLD:
                # Student listening
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 215, 0), 2)
                cv2.putText(frame, "Student Listening", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)

        for person in persons:
            x1, y1, x2, y2 = person['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("Teacher and Student Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
