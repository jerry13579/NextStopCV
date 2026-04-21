import csv
import os
from glob import glob

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

video_folder = "data"
output_folder = "data/annotated/qwen_vlm_seats"
sample_path = "data/sample bus seat.png"
prompt = "Concisely describe the seating situation on the bus"
model_id = "Qwen/Qwen3-VL-2B-Instruct"
sample_seconds = 5
os.makedirs(output_folder, exist_ok=True)
device = "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to(device).eval()


def resize_image(image, max_side=448):
    scale = max_side / max(image.size)
    return image.resize((int(image.width * scale), int(image.height * scale))) if scale < 1 else image


def ask_qwen(frame):
    sample = resize_image(Image.fromarray(cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)))
    image = resize_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": f"The first image is a sample of the bus seat to look for. In the second image, answer this passenger question: {prompt} Reply in 5 words or fewer."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[sample, image], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=12, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()


def put_answer(frame, answer):
    text = answer if answer else "Checking seats..."
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    cv2.putText(frame, text, (frame.shape[1] - size[0] - 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


rows = []
for video_path in glob(f"{video_folder}/*.*"):
    if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
        continue

    name = os.path.splitext(os.path.basename(video_path))[0]
    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(f"{output_folder}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_number, answer, interval = 0, "", max(1, int(fps * sample_seconds))

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        if frame_number % interval == 0:
            answer = ask_qwen(frame)
            rows.append([name, frame_number, round(frame_number / fps, 2), answer])
            print(f"{name} {frame_number}: {answer}")
        put_answer(frame, answer)
        writer.write(frame)
        frame_number += 1

    camera.release()
    writer.release()
    print(f"wrote {output_folder}/{name}.mp4")

with open(f"{output_folder}/summary.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video", "frame", "time_seconds", "answer"])
    writer.writerows(rows)
