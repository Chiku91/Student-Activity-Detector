# modules/standing.py

import torch
import cv2

def run():
    # Load YOLOv5s model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    cap = cv2.VideoCapture(0)
    frame_counter = 0

    def is_standing(box, threshold=1.3):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width == 0:
            return False
        return (height / width) > threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        standing_count = 0

        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # class 0 = person
                if is_standing(box):
                    standing_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Standing", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show count and warning
        cv2.putText(frame, f'Standing Students: {standing_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if standing_count > 0 and (frame_counter // 15) % 2 == 0:
            cv2.putText(frame, "Please don't stand in class; sit down", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        frame_counter += 1
        cv2.imshow("Standing Students Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
