import os
from glob import glob

import cv2
from ultralytics import YOLO

video_folder = "data"
output_folder = "data/annotated/yolo26_empty_seats"
model_path = "runs/detect/data/runs/yolo26_empty_seats/train-2/weights/best.pt"
classes = ["empty_seat", "occupied_seat", "seat_back", "seat_back_occupied"]
colors = [(52, 199, 89), (48, 59, 255), (255, 132, 10), (10, 214, 255)]
score_threshold = 0.25
active_classes = [0]
model = YOLO(model_path)
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
    while True:
        ok, image = camera.read()
        if not ok:
            break
        frame_number += 1
        result = model.predict(image, imgsz=960, conf=score_threshold, classes=active_classes, verbose=False, device="mps")[0]
        boxes = result.boxes
        counts = [0] * len(active_classes)
        if boxes is not None:
            xyxy, cls, scores = boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), class_id, score in zip(xyxy, cls, scores):
                active_index = active_classes.index(class_id)
                counts[active_index] += 1
                color = colors[class_id]
                label = f"{classes[class_id]} {score:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(image, label, (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(image, (12, 12), (560, 42 + 34 * len(classes)), (0, 0, 0), -1)
        for i, class_id in enumerate(active_classes):
            cv2.putText(image, f"{classes[class_id]}: {counts[i]}", (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id], 2)
        cv2.putText(image, f"conf: {score_threshold:.2f}", (20, 40 + 34 * len(active_classes)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(image)
        print(f"{os.path.basename(video_path)} frame {frame_number}", end="\r")
    camera.release()
    writer.release()
    print(f"wrote {output_path}".ljust(80))
