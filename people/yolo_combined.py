import os
from glob import glob

import cv2
import numpy as np
from ultralytics import YOLO

video_folder = "data"
output_folder = "data/annotated/yolo_combined"
people_model = YOLO("yolo11n.pt")
seats_model = YOLO("runs/yolo26_empty_seats/train_all_current_labels/weights/best.pt")
os.makedirs(output_folder, exist_ok=True)

for video_path in glob(f"{video_folder}/*.*"):
    if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
        continue

    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_number = 0
    people_stats = []
    seats_stats = []

    while True:
        ok, image = camera.read()
        if not ok:
            break

        frame_number += 1
        
        # People detection with tracking
        people_result = people_model.track(image, persist=frame_number > 1, tracker="bytetrack.yaml", classes=[0], conf=0.25, verbose=False)[0]
        people_boxes = people_result.boxes.xyxy.cpu().numpy() if people_result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        ids = people_result.boxes.id.cpu().numpy().astype(int) if people_result.boxes is not None and people_result.boxes.id is not None else np.arange(1, len(people_boxes) + 1)
        
        # Calculate head areas and rolling average
        head_areas = np.maximum((people_boxes[:, 2] - people_boxes[:, 0]) * (people_boxes[:, 3] - people_boxes[:, 1]) * 0.12, 1) if len(people_boxes) else np.array([0.0])
        people_count = len(people_boxes)
        people_stats = (people_stats + [(people_count, head_areas.mean() if len(head_areas) > 0 else 0)])[-int(fps * 5):]
        people_avg, head_area_avg = np.mean(people_stats, axis=0)
        
        density = float(people_avg * head_area_avg / (height * width))
        crowdedness_pct = min(100, round(100 * density / 0.155, 1))

        # Seat detection
        seats_result = seats_model.predict(image, imgsz=960, conf=0.25, classes=[0], verbose=False, device=0)[0]
        seats_boxes = seats_result.boxes
        seat_count = 0
        if seats_boxes is not None:
            xyxy = seats_boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                seat_count += 1
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (52, 199, 89), 2)
        seats_stats = (seats_stats + [seat_count])[-int(fps * 5):]
        seat_count_avg = round(np.mean(seats_stats))

        # Draw people bounding boxes
        for track_id, (x1, y1, x2, y2) in zip(ids, people_boxes):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, f"Person {track_id}", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Top left: People metrics
        cv2.rectangle(image, (12, 12), (380, 130), (0, 0, 0), -1)
        cv2.putText(image, f"{crowdedness_pct:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
        cv2.putText(image, f"Headcount: {round(people_avg)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Density: {density:.4f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Head area: {head_area_avg:.1f}px", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Top right: Seat metrics
        cv2.rectangle(image, (width - 280, 12), (width - 12, 80), (0, 0, 0), -1)
        cv2.putText(image, f"Empty seats: {seat_count_avg}", (width - 270, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 199, 89), 2)
        cv2.putText(image, f"5s rolling avg", (width - 270, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 199, 89), 2)

        writer.write(image)
        print(f"{os.path.basename(video_path)} frame {frame_number}", end="\r")

    camera.release()
    writer.release()
    print(f"wrote {output_path}".ljust(80))
