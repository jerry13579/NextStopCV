import logging
import os
from glob import glob

import cv2

from bus_scene_analysis import config
from edge_compute.scene_analyzer import EdgeSceneAnalyzer

video_folder = "data"
output_folder = config.OUTPUT_VIDEO_DIR
box_color = (255, 0, 0)
seat_color = (52, 199, 89)


def run():
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(message)s")
    analyzer = EdgeSceneAnalyzer(enable_llava=config.USE_LLAVA)
    for video_path in glob(f"{video_folder}/*.*"):
        if not video_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".m4v")):
            continue
        analyze_video(video_path, analyzer)


def analyze_video(video_path, analyzer):
    analyzer.reset()
    camera = cv2.VideoCapture(video_path)
    fps = camera.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{output_folder}/{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_number, summary, backend_interval = 0, "", max(1, int(fps * config.BACKEND_SAMPLE_SECONDS))
    while True:
        ok, image = camera.read()
        if not ok:
            break
        frame_number += 1
        analysis = analyzer.analyze_frame(image, frame_id=f"{os.path.basename(video_path)}:{frame_number}", fps=fps, include_image=config.LLAVA_INCLUDE_IMAGE, track=frame_number > 1, run_backend=frame_number % backend_interval == 1)
        summary = draw_analysis(image, analysis, summary)
        writer.write(image)
        print(f"{os.path.basename(video_path)} frame {frame_number}", end="\r")
    camera.release()
    writer.release()
    print(f"wrote {output_path}".ljust(80))


def draw_analysis(image, analysis, summary=""):
    metrics = analysis["yolo_metrics"]
    summary = analysis["llava_summary"] or summary or "Waiting for backend LLM summary..."
    h, w = image.shape[:2]
    for detection in analysis["detections"]:
        x1, y1, x2, y2 = detection["bbox"]
        color = box_color if detection["class"] == "person" else seat_color
        label = f"Person {detection.get('track_id', '')}".strip() if detection["class"] == "person" else f"Empty seats {detection['confidence']:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(image, (12, 12), (410, 130), (0, 0, 0), -1)
    cv2.putText(image, f"{metrics['crowdedness_pct']}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
    cv2.putText(image, f"People: {metrics['passenger_count']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Density: {metrics['density']:.4f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Head area: {metrics['avg_head_area_px']:.1f}px", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.rectangle(image, (w - 330, 12), (w - 12, 94), (0, 0, 0), -1)
    cv2.putText(image, f"Empty seats: {metrics['empty_seat_count']}", (w - 320, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, seat_color, 2)
    cv2.putText(image, f"Crowding: {metrics['crowding_level']}", (w - 320, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, seat_color, 2)
    draw_bottom_message(image, summary)
    return summary


def draw_bottom_message(image, text):
    h, w = image.shape[:2]
    lines = wrap_text(text, w - 56, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[:4]
    top = h - 32 - 28 * len(lines)
    cv2.rectangle(image, (12, max(0, top - 14)), (w - 12, h - 12), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (22, top + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def wrap_text(text, max_width, font, scale, thickness):
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if cv2.getTextSize(test, font, scale, thickness)[0][0] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    return lines + ([line] if line else [])


if __name__ == "__main__":
    run()
