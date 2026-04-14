import sys

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
camera_source = sys.argv[1] if len(sys.argv) > 1 else "0"
camera = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)


def box_overlap(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def split_benches(chair_boxes):
    split_boxes = []
    for x1, y1, x2, y2 in chair_boxes:
        width, height = x2 - x1, y2 - y1
        seats = max(1, round(width / max(height * 0.9, 1)))
        seat_width = width / seats
        split_boxes.extend([[x1 + i * seat_width, y1, x1 + (i + 1) * seat_width, y2] for i in range(seats)])
    return np.array(split_boxes, dtype=np.float32) if split_boxes else np.empty((0, 4), dtype=np.float32)


while True:
    ok, image = camera.read()
    if not ok:
        break

    image_height, image_width = image.shape[:2]
    result = model(image, verbose=False, classes=[0, 56])[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    person_boxes = boxes[classes == 0]
    chair_boxes = split_benches(boxes[classes == 56])
    head_areas = np.maximum(
        (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]) * 0.12,
        1,
    ) if len(person_boxes) else np.array([0.0])
    density = float(len(person_boxes) * head_areas.mean() / (image_height * image_width))
    crowdedness_pct = min(100, round(100 * density / 0.155, 1))
    occupied_chairs = [any(box_overlap(chair_box, person_box) > 0 for person_box in person_boxes) for chair_box in chair_boxes]
    occupied_count = sum(occupied_chairs)

    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, "Person", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for (x1, y1, x2, y2), is_occupied in zip(chair_boxes, occupied_chairs):
        color = (0, 255, 0) if is_occupied else (0, 0, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, "Chair occupied" if is_occupied else "Chair empty", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(image, f"Crowdedness: {crowdedness_pct}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"People: {len(person_boxes)}  Chairs: {len(chair_boxes)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Density: {density:.4f}  Avg head px: {head_areas.mean():.1f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Chairs occupied: {occupied_count}/{len(chair_boxes)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Bus Crowdedness", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
