import csv
import os
from glob import glob

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

video_folder = "data"
output_folder = "data/annotated/grounding_dino_seats"
prompt = "empty green bus seats with silver outline front visible"
model_id = "IDEA-Research/grounding-dino-tiny"
box_threshold = 0.25
text_threshold = 0.25
sample_seconds = 5
os.makedirs(output_folder, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()


def detect(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=[[prompt]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
    return result["boxes"].cpu().numpy(), result["scores"].cpu().numpy()


def put_count(frame, count):
    text = f"Seats: {count}"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
    cv2.putText(frame, text, (frame.shape[1] - size[0] - 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)


rows = []
for video_path in glob(f"{video_folder}/*.*"):
    if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
        continue

    name = os.path.splitext(os.path.basename(video_path))[0]
    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(f"{output_folder}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_number, count, interval = 0, 0, max(1, int(fps * sample_seconds))

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        if frame_number % interval == 0:
            boxes, scores = detect(frame)
            count = len(boxes)
            rows.append([name, frame_number, round(frame_number / fps, 2), count, ";".join(f"{score:.3f}" for score in scores)])
            print(f"{name} {frame_number}: {count}")
        put_count(frame, count)
        writer.write(frame)
        frame_number += 1

    camera.release()
    writer.release()
    print(f"wrote {output_folder}/{name}.mp4")

with open(f"{output_folder}/summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video", "frame", "time_seconds", "seat_count", "scores"])
    writer.writerows(rows)
