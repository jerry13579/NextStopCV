import cv2
import numpy as np
from ultralytics import YOLO

# Hard-code the same YOLO11 model and crowdedness scaling as the image script.
model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

while True:
    ok, image = camera.read()
    if not ok:
        break

    # Detect people in the current webcam frame, then use each person box as a proxy for one head.
    image_height, image_width = image.shape[:2]
    person_boxes = model(image, verbose=False, classes=[0])[0].boxes.xyxy.cpu().numpy()

    # Approximate visible head size as a fixed fraction of each person box area.
    head_areas = np.maximum(
        (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]) * 0.12,
        1,
    ) if len(person_boxes) else np.array([0.0])

    # Combine head count and average head size into a simple density score,
    # then scale that density into a 0-100 crowdedness percentage.
    density = float(len(person_boxes) * head_areas.mean() / (image_height * image_width))
    crowdedness_pct = min(100, round(100 * density / 0.16, 1))

    cv2.putText(image, f"Crowdedness: {crowdedness_pct}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Heads: {len(person_boxes)}  Density: {density:.4f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Bus Crowdedness", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
