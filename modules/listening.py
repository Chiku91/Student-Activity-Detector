# modules/listening.py

import cv2
import mediapipe as mp
import time
import numpy as np
import random

def run():
    # Face mesh for attention tracking
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(0)

    attention_start_time = None
    attention_duration = 0
    last_detected = False
    ATTENTION_THRESHOLD = 20
    MOVEMENT_THRESHOLD = 5000
    prev_gray = None

    # Warning animation
    warning_text = "Concentrate! Listen carefully and ask doubts!"
    animation_frames = []
    for i in range(15):
        frame = np.zeros((80, 800, 3), dtype=np.uint8)
        color = (0, 0, 255 - i * 15) if i < 8 else (0, 0, (i - 7) * 15)
        cv2.putText(frame, warning_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        animation_frames.append(frame)
    animation_index = 0

    last_visual_switch = time.time()
    VISUAL_SWITCH_INTERVAL = 2
    visual_index = 0

    # Visual generators
    def generate_bar_chart():
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        for i in range(5):
            h = random.randint(50, 300)
            x = 50 + i * 100
            y = 350 - h
            cv2.rectangle(img, (x, y), (x + 80, 350), (0, 128, 255), -1)
            cv2.putText(img, str(h), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "Bar Chart Example", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        return img

    def generate_line_graph():
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        points = [(x * 50 + 50, 350 - random.randint(30, 250)) for x in range(10)]
        for i in range(1, len(points)):
            cv2.line(img, points[i - 1], points[i], (0, 0, 255), 2)
        for p in points:
            cv2.circle(img, p, 5, (255, 0, 0), -1)
        cv2.putText(img, "Line Graph Example", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        return img

    def generate_pie_chart():
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        center = (300, 200)
        radius = 100
        angles = [random.randint(30, 120) for _ in range(3)]
        total = sum(angles)
        start = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]
        for a, c in zip(angles, colors):
            end = start + int(a / total * 360)
            cv2.ellipse(img, center, (radius, radius), 0, start, end, c, -1)
            start = end
        cv2.circle(img, center, radius, (0, 0, 0), 2)
        cv2.putText(img, "Pie Chart Example", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        return img

    def generate_equation_visual():
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        equations = [
            "E = mc^2", "a^2 + b^2 = c^2", "F = ma",
            "V = IR", "y = mx + b", "sin^2θ + cos^2θ = 1"
        ]
        eq = random.choice(equations)
        cv2.putText(img, "Math Concept", (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(img, eq, (150, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
        return img

    visual_generators = [generate_bar_chart, generate_line_graph, generate_pie_chart, generate_equation_visual]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        current_time = time.time()
        status = "Not attentive"
        movement_detected = False

        if prev_gray is not None:
            frame_diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            movement_amount = np.sum(thresh)
            if movement_amount > MOVEMENT_THRESHOLD:
                movement_detected = True
        prev_gray = gray.copy()

        show_warning = False

        if result.multi_face_landmarks and not movement_detected:
            if not last_detected:
                attention_start_time = current_time
                attention_duration = 0
            else:
                attention_duration = current_time - attention_start_time

            if attention_duration >= ATTENTION_THRESHOLD:
                status = "Attentive"
            else:
                status = "Listening to lesson"
            last_detected = True
        else:
            attention_start_time = None
            attention_duration = 0
            last_detected = False
            show_warning = True
            if movement_detected:
                status = "Not concentrating"
            else:
                status = "Not attentive"

        color = (0, 255, 0) if status == "Attentive" else (0, 0, 255)
        cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if show_warning:
            anim_frame = animation_frames[animation_index]
            animation_index = (animation_index + 1) % len(animation_frames)
            anim_resized = cv2.resize(anim_frame, (min(w - 20, 700), 80))
            y_offset = h - 90
            x_offset = (w - anim_resized.shape[1]) // 2
            frame[y_offset:y_offset + anim_resized.shape[0], x_offset:x_offset + anim_resized.shape[1]] = anim_resized

        # Main webcam frame
        cv2.imshow("Attention Detector", frame)

        # Concept visual rotation
        if current_time - last_visual_switch > VISUAL_SWITCH_INTERVAL:
            visual_index = (visual_index + 1) % len(visual_generators)
            last_visual_switch = current_time
        visual_img = visual_generators[visual_index]()
        cv2.imshow("Concept Visuals", visual_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
