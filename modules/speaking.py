import cv2
import mediapipe as mp
import torch
import numpy as np
import sounddevice as sd
import queue
import time

def run():
    # Load YOLOv5s model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5)

    # Webcam
    cap = cv2.VideoCapture(0)

    # Audio processing
    audio_queue = queue.Queue()
    VOLUME_THRESHOLD = 0.0006
    current_volume = 0.0

    # Speaking detection state
    speaking_start_time = None
    SPEAKING_DURATION_THRESHOLD = 5  # in seconds
    speaking_detected = False

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

        # Get volume
        while not audio_queue.empty():
            current_volume = audio_queue.get()

        audio_active = current_volume > VOLUME_THRESHOLD

        # Detect people with YOLOv5
        results = model(rgb)
        people = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                people.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Face mesh detection
        face_results = face_mesh.process(rgb)
        face_found = False
        speaking_students = 0

        if face_results.multi_face_landmarks:
            face_found = True
            for landmarks in face_results.multi_face_landmarks:
                nose = landmarks.landmark[1]
                cx, cy = int(nose.x * w), int(nose.y * h)

                # Track speaking time
                if audio_active:
                    if speaking_start_time is None:
                        speaking_start_time = time.time()
                    elif time.time() - speaking_start_time >= SPEAKING_DURATION_THRESHOLD:
                        speaking_detected = True
                else:
                    speaking_start_time = None
                    speaking_detected = False

                if speaking_detected:
                    speaking_students += 1
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "Student is speaking", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Reset state if no face found
        if not face_found:
            speaking_start_time = None
            speaking_detected = False

        # Display information
        cv2.putText(frame, f"Speaking Students: {speaking_students}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Audio Volume: {current_volume:.4f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Speaking Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    stream.close()
    cap.release()
    cv2.destroyAllWindows()
