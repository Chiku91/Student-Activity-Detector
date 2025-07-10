import cv2
import mediapipe as mp
import torch
import numpy as np
import sounddevice as sd
import queue

def run():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    # Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5)

    # Webcam
    cap = cv2.VideoCapture(0)

    # Audio
    audio_queue = queue.Queue()
    VOLUME_THRESHOLD = 0.0005
    current_volume = 0.0

    def audio_callback(indata, frames, time_info, status):
        volume_norm = np.linalg.norm(indata) / frames
        audio_queue.put(volume_norm)

    # Start audio stream
    stream = sd.InputStream(channels=1, samplerate=44100, callback=audio_callback)
    stream.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get audio volume from queue
        while not audio_queue.empty():
            current_volume = audio_queue.get()

        audio_active = current_volume > VOLUME_THRESHOLD

        # Detect people with YOLOv5 (for green bounding boxes)
        results = model(rgb)
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 = person
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Face Mesh detection
        face_results = face_mesh.process(rgb)
        speaking_students = 0

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                cx, cy = int(nose.x * w), int(nose.y * h)

                if audio_active:
                    speaking_students += 1
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "Speaking", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                    cv2.putText(frame, "Silent", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display info
        cv2.putText(frame, f"Speaking Students: {speaking_students}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Audio Volume: {current_volume:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Speaking Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
