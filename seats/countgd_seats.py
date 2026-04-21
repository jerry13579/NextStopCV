import csv
import os
from glob import glob

import cv2
from gradio_client import Client, handle_file

video_folder = "data"
output_folder = "data/annotated/countgd_seats"
sample_path = "data/sample bus seat.png"
prompt = "empty green bus seats with silver outline front visible"
sample_seconds = 5
os.makedirs(output_folder, exist_ok=True)
client = Client("nikigoli/countgd", httpx_kwargs={"timeout": 180})


def count_frame(frame, name, frame_number):
    path = f"{output_folder}/{name}_{frame_number}.jpg"
    cv2.imwrite(path, frame)
    _, count = client.predict(handle_file(path), prompt, {"image": handle_file(sample_path), "points": []}, api_name="/count_main")
    os.remove(path)
    return int(round(count["value"]))


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
            count = count_frame(frame, name, frame_number)
            rows.append([name, frame_number, round(frame_number / fps, 2), count, sample_path])
            print(f"{name} {frame_number}: {count}")
        put_count(frame, count)
        writer.write(frame)
        frame_number += 1

    camera.release()
    writer.release()
    print(f"wrote {output_folder}/{name}.mp4")

with open(f"{output_folder}/summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video", "frame", "time_seconds", "seat_count", "sample_image"])
    writer.writerows(rows)
