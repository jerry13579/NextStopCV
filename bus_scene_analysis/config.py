import logging

YOLO_PEOPLE_MODEL = "yolo26n.pt"
YOLO_SEAT_MODEL = "yolo_finetuning/runs/yolo26_empty_seats/train_all_current_labels/weights/best.pt"
YOLO_PERSON_CONF = 0.25
YOLO_SEAT_CONF = 0.25
YOLO_SEAT_IMGSZ = 960
YOLO_DEVICE = 0
YOLO_HALF = True
YOLO_TRACKER = "bytetrack.yaml"
YOLO_ROLLING_SECONDS = 5
CROWDING_DENSITY_FULL = 0.155
USE_LLAVA = True
LLAVA_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
LLAVA_VISION_ENCODER_MODEL_ID = "google/siglip-base-patch16-224"
LLAVA_ENCODER_DEVICE = "cpu"
LLAVA_SERVER_DEVICE = "cuda:0"
LLAVA_MAX_NEW_TOKENS = 48
BACKEND_SAMPLE_SECONDS = 5
LLAVA_INCLUDE_IMAGE = True
OUTPUT_VIDEO_DIR = "output/annotated"
LOG_LEVEL = logging.INFO
