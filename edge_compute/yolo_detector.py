from datetime import datetime
from importlib.util import find_spec

import numpy as np
from ultralytics import YOLO

from bus_scene_analysis import config
from bus_scene_analysis.schemas.detection_schema import detection_item, detection_output


class BusYoloDetector:
    def __init__(self, people_model=None, seat_model=None):
        self.people_model = people_model or YOLO(config.YOLO_PEOPLE_MODEL)
        self.seat_model = seat_model or YOLO(config.YOLO_SEAT_MODEL)
        self.tracking_enabled = find_spec("lap") is not None
        self.reset()

    def reset(self):
        self.people_stats, self.seat_stats, self.crowdedness_stats = [], [], []

    def analyze_frame(self, image, frame_id="0", timestamp=None, fps=30, track=True):
        timestamp = timestamp or datetime.utcnow().isoformat()
        h, w = image.shape[:2]
        try:
            people_result = (self.people_model.track(image, persist=track, tracker=config.YOLO_TRACKER, classes=[0], conf=config.YOLO_PERSON_CONF, verbose=False, device=config.YOLO_DEVICE, half=config.YOLO_HALF) if track and self.tracking_enabled else self.people_model.predict(image, classes=[0], conf=config.YOLO_PERSON_CONF, verbose=False, device=config.YOLO_DEVICE, half=config.YOLO_HALF))[0]
        except ModuleNotFoundError:
            self.tracking_enabled = False
            people_result = self.people_model.predict(image, classes=[0], conf=config.YOLO_PERSON_CONF, verbose=False, device=config.YOLO_DEVICE, half=config.YOLO_HALF)[0]
        people_boxes = people_result.boxes.xyxy.cpu().numpy() if people_result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        people_scores = people_result.boxes.conf.cpu().numpy() if people_result.boxes is not None else np.array([])
        ids = people_result.boxes.id.cpu().numpy().astype(int) if people_result.boxes is not None and people_result.boxes.id is not None else np.arange(1, len(people_boxes) + 1)
        head_areas = np.maximum((people_boxes[:, 2] - people_boxes[:, 0]) * (people_boxes[:, 3] - people_boxes[:, 1]) * 0.12, 1) if len(people_boxes) else np.array([0.0])
        window = max(1, int(fps * config.YOLO_ROLLING_SECONDS))
        self.people_stats = (self.people_stats + [(len(people_boxes), float(head_areas.mean()))])[-window:]
        people_avg, head_area_avg = np.mean(self.people_stats, axis=0)
        density = float(people_avg * head_area_avg / (h * w))
        current_density = float(len(people_boxes) * float(head_areas.mean()) / (h * w))
        current_crowdedness_pct = min(100, 100 * current_density / config.CROWDING_DENSITY_FULL)
        self.crowdedness_stats = (self.crowdedness_stats + [current_crowdedness_pct])[-window:]
        crowdedness_pct = int(round(float(np.mean(self.crowdedness_stats)) / 5) * 5)

        seat_result = self.seat_model.predict(image, imgsz=config.YOLO_SEAT_IMGSZ, conf=config.YOLO_SEAT_CONF, classes=[0], verbose=False, device=config.YOLO_DEVICE, half=config.YOLO_HALF)[0]
        seat_boxes = seat_result.boxes.xyxy.cpu().numpy() if seat_result.boxes is not None else np.empty((0, 4), dtype=np.float32)
        seat_scores = seat_result.boxes.conf.cpu().numpy() if seat_result.boxes is not None else np.array([])
        self.seat_stats = (self.seat_stats + [len(seat_boxes)])[-window:]
        empty_seat_count = int(round(float(np.mean(self.seat_stats)))) if self.seat_stats else 0

        detections = [detection_item("person", score, box, track_id) for box, score, track_id in zip(people_boxes, people_scores, ids)]
        detections += [detection_item("empty_seat", score, box) for box, score in zip(seat_boxes, seat_scores)]
        metrics = {
            "passenger_count": int(round(float(people_avg))),
            "current_passenger_count": int(len(people_boxes)),
            "empty_seat_count": empty_seat_count,
            "current_empty_seat_count": int(len(seat_boxes)),
            "occupied_seat_count": None,
            "standing_passenger_count": None,
            "wheelchair_area_occupancy": None,
            "door_area_congestion": None,
            "density": round(density, 6),
            "crowdedness_pct": crowdedness_pct,
            "crowding_level": self.crowding_level(crowdedness_pct),
            "avg_head_area_px": round(float(head_area_avg), 2),
        }
        return detection_output(timestamp, frame_id, detections, metrics)

    def crowding_level(self, pct):
        if pct >= 70:
            return "high"
        if pct >= 35:
            return "moderate"
        return "low"
