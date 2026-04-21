import os
from glob import glob

import cv2
import numpy as np
from ultralytics import YOLO

video_folder = "data"
output_folder = "data/annotated/yolo11_tracking"
model = YOLO("yolo11n.pt")
os.makedirs(output_folder, exist_ok=True)
box_color = (255, 0, 0)


for video_path in glob(f"{video_folder}/*.*"):
    if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
        continue

    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_number = 0
    stats = []

    while True:
        ok, image = camera.read()
        if not ok:
            break

        frame_number += 1
        result = model.track(image, persist=frame_number > 1, tracker="bytetrack.yaml", classes=[0], conf=0.25, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes is not None and result.boxes.id is not None else np.arange(1, len(boxes) + 1)
        head_areas = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * 0.12, 1) if len(boxes) else np.array([0.0])
        stats = (stats + [(len(boxes), head_areas.mean() if len(head_areas) > 0 else 0)])[-int(fps * 5):]
        people_avg, head_area_avg = np.mean(stats, axis=0)
        density_avg = float(people_avg * head_area_avg / (height * width))
        crowdedness_avg = min(100, round(100 * density_avg / 0.155, 1))

        for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(image, f"Person {track_id}", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.rectangle(image, (12, 12), (380, 130), (0, 0, 0), -1)
        cv2.putText(image, f"{crowdedness_avg:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
        cv2.putText(image, f"Headcount: {round(people_avg)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Density: {density_avg:.4f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Head area: {head_area_avg:.1f}px", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(image)
        print(f"{os.path.basename(video_path)} frame {frame_number}", end="\r")

    camera.release()
    writer.release()
    print(f"wrote {output_path}".ljust(80))
