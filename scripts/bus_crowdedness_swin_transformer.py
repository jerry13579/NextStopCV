import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

model_id = "facebook/mask2former-swin-small-coco-instance"
processor = AutoImageProcessor.from_pretrained(model_id)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
camera_id = sys.argv[1] if len(sys.argv) > 1 else "swin_transformer"
output_dir = f"data/annotated/{camera_id}"


def mask_box(mask):
    y, x = np.where(mask)
    return [x.min(), y.min(), x.max(), y.max()]


def split_benches(chair_boxes):
    split_boxes = []
    for x1, y1, x2, y2 in chair_boxes:
        width, height = x2 - x1, y2 - y1
        seats = max(1, round(width / max(height * 0.9, 1)))
        seat_width = width / seats
        split_boxes.extend([[x1 + i * seat_width, y1, x1 + (i + 1) * seat_width, y2] for i in range(seats)])
    return np.array(split_boxes, dtype=np.float32) if split_boxes else np.empty((0, 4), dtype=np.float32)


def sorted_boxes(boxes):
    return boxes[np.lexsort((boxes[:, 0], boxes[:, 1]))] if len(boxes) else boxes


def person_points(person_boxes):
    return [((x1 + x2) / 2, y1 + (y2 - y1) * 0.55) for x1, y1, x2, y2 in person_boxes]


def seat_region(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return x1 - width * 0.1, y1 - height * 0.75, x2 + width * 0.1, y2 + height * 0.15


def occupied_seats(chair_boxes, person_boxes):
    return [any(sx1 <= px <= sx2 and sy1 <= py <= sy2 for px, py in person_points(person_boxes)) for sx1, sy1, sx2, sy2 in [seat_region(box) for box in chair_boxes]]


def segments(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    result = processor.post_process_instance_segmentation(output, target_sizes=[image.size[::-1]], threshold=0.5)[0]
    segmentation = result["segmentation"].cpu().numpy()
    masks = [(info["label_id"], segmentation == info["id"]) for info in result["segments_info"] if info["label_id"] in [0, 56] and (segmentation == info["id"]).any()]
    return [(label, mask, mask_box(mask)) for label, mask in masks]


os.makedirs(output_dir, exist_ok=True)

for image_path in sorted(glob.glob("data/*.jpg") + glob.glob("data/*.png")):
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    detected = segments(image_path)
    person_masks = [mask for label, mask, box in detected if label == 0]
    person_boxes = np.array([box for label, mask, box in detected if label == 0], dtype=np.float32)
    chair_masks = [mask for label, mask, box in detected if label == 56]
    chair_boxes = sorted_boxes(split_benches(np.array([box for label, mask, box in detected if label == 56], dtype=np.float32)))
    head_areas = np.maximum((person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]) * 0.12, 1) if len(person_boxes) else np.array([0.0])
    density = float(len(person_boxes) * head_areas.mean() / (image_height * image_width))
    crowdedness_pct = min(100, round(100 * density / 0.155, 1))
    occupied_chairs = occupied_seats(chair_boxes, person_boxes)
    occupied_count = sum(occupied_chairs)

    for mask in person_masks:
        image[mask] = image[mask] * 0.55 + np.array([255, 0, 0]) * 0.45
    for mask in chair_masks:
        image[mask] = image[mask] * 0.55 + np.array([0, 0, 255]) * 0.45
    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, "Person", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.circle(image, (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.55)), 4, (255, 255, 0), -1)
    for (x1, y1, x2, y2), is_occupied in zip(chair_boxes, occupied_chairs):
        color = (0, 255, 0) if is_occupied else (0, 0, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, "Chair occupied" if is_occupied else "Chair empty", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(image, f"Crowdedness: {crowdedness_pct}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"People: {len(person_boxes)}  Chairs: {len(chair_boxes)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f"Density: {density:.4f}  Avg head px: {head_areas.mean():.1f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, f"Chairs occupied: {occupied_count}/{len(chair_boxes)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(f"{output_dir}/{os.path.basename(image_path)}", image)

    print({
        "file": os.path.basename(image_path),
        "camera_id": camera_id,
        "heads": int(len(person_boxes)),
        "avg_head_px": round(float(head_areas.mean()), 1),
        "density": round(density, 4),
        "crowdedness_pct": crowdedness_pct,
        "chairs": int(len(chair_boxes)),
        "occupied_chairs": int(occupied_count),
        "empty_chairs": int(len(chair_boxes) - occupied_count),
        "annotated_output": f"{output_dir}/{os.path.basename(image_path)}",
    })
