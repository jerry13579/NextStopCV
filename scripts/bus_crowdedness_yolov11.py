import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Hard-code a small YOLO11 model and scan every sample image in data/.
model = YOLO("yolo11n.pt")

for image_path in sorted(glob.glob("data/*.png")):
    # Detect people in the bus image, then use each person box as a proxy for one head.
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    person_boxes = model(image, verbose=False, classes=[0])[0].boxes.xyxy.cpu().numpy()

    # Approximate visible head size as a fixed fraction of each person box area.
    if len(person_boxes):
        # Compute head area as 12% of person box area
        raw_head_areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]) * 0.12
        # Clip minimum head area as 1 pixel
        head_areas = np.maximum(raw_head_areas, 1)
    else:
        # Fallback if there are no heads detected
        np.array([0.0])

    # Combine head count and average head size into a simple density score,
    # then scale that density into a 0-100 crowdedness percentage.
    density = float(len(person_boxes) * head_areas.mean() / (image_height * image_width))
    crowdedness_pct = min(100, round(100 * density / 0.155, 1))

    print({
        "file": os.path.basename(image_path),
        "heads": int(len(person_boxes)),
        "avg_head_px": round(float(head_areas.mean()), 1),
        "density": round(density, 4),
        "crowdedness_pct": crowdedness_pct,
    })
