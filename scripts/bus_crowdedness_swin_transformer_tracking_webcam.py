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
camera_source = sys.argv[1] if len(sys.argv) > 1 else "0"
camera = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)
seats, tracks, next_track_id = [], [], 1
prev_gray = None
min_seat_area, min_seat_width, min_seat_height = 1800, 35, 25


def mask_box(mask):
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def split_benches(chair_boxes):
    split_boxes = []
    for x1, y1, x2, y2 in chair_boxes:
        width, height = x2 - x1, y2 - y1
        seats_count = max(1, round(width / max(height * 1.15, 1))) if width / max(height, 1) > 2.1 else 1
        seat_width = width / seats_count
        split_boxes.extend([[x1 + i * seat_width, y1, x1 + (i + 1) * seat_width, y2] for i in range(seats_count)])
    return np.array(split_boxes, dtype=np.float32) if split_boxes else np.empty((0, 4), dtype=np.float32)


def center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def valid_seat(box):
    return area(box) >= min_seat_area and box[2] - box[0] >= min_seat_width and box[3] - box[1] >= min_seat_height


def overlap(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def merge_boxes(boxes, overlap_threshold=0.45, distance_threshold=0.35):
    merged = []
    for box in sorted(boxes, key=area, reverse=True):
        match = next((i for i, merged_box in enumerate(merged) if overlap(box, merged_box) / max(min(area(box), area(merged_box)), 1) > overlap_threshold or np.linalg.norm(center(box) - center(merged_box)) < max(box[2] - box[0], box[3] - box[1]) * distance_threshold), None)
        if match is None:
            merged.append(box)
        else:
            merged[match] = np.array([min(merged[match][0], box[0]), min(merged[match][1], box[1]), max(merged[match][2], box[2]), max(merged[match][3], box[3])], dtype=np.float32)
    return np.array(merged, dtype=np.float32) if merged else np.empty((0, 4), dtype=np.float32)


def clean_seats():
    kept = []
    for seat in sorted(seats, key=lambda seat: area(seat["box"]), reverse=True):
        duplicate = any(overlap(seat["box"], other["box"]) / max(min(area(seat["box"]), area(other["box"])), 1) > 0.6 for other in kept)
        if seat["occupied"] or valid_seat(seat["box"]) and not duplicate:
            kept.append(seat)
    seats[:] = kept


def filter_seats(boxes):
    filtered = [box for box in boxes if valid_seat(box)]
    return np.array(filtered, dtype=np.float32) if filtered else np.empty((0, 4), dtype=np.float32)


def person_point(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y1 + (y2 - y1) * 0.55], dtype=np.float32)


def box_height(box):
    return box[3] - box[1]


def seat_region(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return np.array([x1 - width * 0.15, y1 - height * 0.8, x2 + width * 0.15, y2 + height * 0.2], dtype=np.float32)


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def transform_box(box, matrix, image_width, image_height):
    width, height = box[2] - box[0], box[3] - box[1]
    moved = cv2.transform(center(box).reshape(1, 1, 2), matrix).reshape(2)
    return np.array([(moved[0] - width / 2).clip(0, image_width - 1), (moved[1] - height / 2).clip(0, image_height - 1), (moved[0] + width / 2).clip(0, image_width - 1), (moved[1] + height / 2).clip(0, image_height - 1)], dtype=np.float32)


def compensate_camera_motion(gray, image_width, image_height):
    global prev_gray
    if prev_gray is None:
        prev_gray = gray
        return
    points = cv2.goodFeaturesToTrack(prev_gray, 250, 0.01, 8)
    if points is not None:
        moved, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
        good = status.ravel() == 1
        matrix, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=3) if good.sum() > 12 else (None, None)
        if matrix is not None and inliers.sum() > 12:
            for seat in seats:
                seat["box"] = transform_box(seat["box"], matrix, image_width, image_height)
            for track in tracks:
                track["box"] = transform_box(track["box"], matrix, image_width, image_height)
                track["point"] = cv2.transform(track["point"].reshape(1, 1, 2), matrix).reshape(2)
                track["history"] = [(cv2.transform(point.reshape(1, 1, 2), matrix).reshape(2), height) for point, height in track["history"]]
    prev_gray = gray


def frame_segments(image):
    frame = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    result = processor.post_process_instance_segmentation(output, target_sizes=[frame.size[::-1]], threshold=0.7)[0]
    segmentation = result["segmentation"].cpu().numpy()
    masks = [(info["label_id"], segmentation == info["id"]) for info in result["segments_info"] if info["label_id"] in [0, 56] and (segmentation == info["id"]).any()]
    return [(label, mask, mask_box(mask)) for label, mask in masks]


def update_seats(chair_boxes):
    for seat in seats:
        seat["seen"] = False
    for box in chair_boxes:
        match = min(seats, key=lambda seat: np.linalg.norm(center(seat["box"]) - center(box)), default=None)
        if match and (inside(center(box), seat_region(match["box"])) or overlap(box, match["box"]) / max(area(box), 1) > 0.35 or np.linalg.norm(center(match["box"]) - center(box)) < max(box[2] - box[0], box[3] - box[1]) * 0.8):
            if not match["occupied"] and 0.25 < area(box) / max(area(match["box"]), 1) < 2.0:
                match["box"] = match["box"] * 0.9 + box * 0.1
            match["seen"], match["missed"] = True, 0
        elif any(seat["occupied"] and inside(center(box), seat_region(seat["box"])) for seat in seats):
            pass
        elif valid_seat(box):
            seats.append({"box": box, "seen": True, "missed": 0, "occupied": False, "occupant": None, "away": 0})
    for seat in seats:
        seat["missed"] += 0 if seat["seen"] else 1
    seats[:] = [seat for seat in seats if seat["missed"] < 240 or seat["occupied"]]
    clean_seats()


def update_tracks(person_boxes):
    global next_track_id
    for track in tracks:
        track["matched"], track["missed"] = False, track["missed"] + 1
    for box in person_boxes:
        point = person_point(box)
        candidates = [track for track in tracks if not track["matched"]]
        match = min(candidates, key=lambda track: np.linalg.norm(track["point"] - point) - overlap(track["box"], box) * 0.02, default=None)
        if match and (np.linalg.norm(match["point"] - point) < max(box[2] - box[0], box[3] - box[1], 120) or overlap(match["box"], box) / max(min(area(match["box"]), area(box)), 1) > 0.25):
            speed = np.linalg.norm(match["point"] - point)
            match.update({"box": box, "point": point, "matched": True, "missed": 0, "still": match["still"] + 1 if speed < 22 else 0})
            match["history"] = (match["history"] + [(point, box_height(box))])[-12:]
        else:
            tracks.append({"id": next_track_id, "box": box, "point": point, "matched": True, "missed": 0, "still": 0, "seat": None, "dwell": 0, "history": [(point, box_height(box))]})
            next_track_id += 1
    tracks[:] = [track for track in tracks if track["missed"] < 24]


def sat_down(track, seat):
    if len(track["history"]) < 4:
        return False
    points = np.array([point for point, height in track["history"]], dtype=np.float32)
    heights = np.array([height for point, height in track["history"]], dtype=np.float32)
    x1, y1, x2, y2 = seat_region(seat["box"])
    start, end = points[0], points[-1]
    drop = end[1] - points[: len(points) // 2].mean(axis=0)[1]
    horizontal = abs(end[0] - start[0])
    compressed = heights[-1] < heights[: len(heights) // 2].mean() * 0.9
    return inside(end, [x1, y1, x2, y2]) and drop > max((y2 - y1) * 0.12, 12) and horizontal < (x2 - x1) * 0.9 or inside(end, [x1, y1, x2, y2]) and compressed and drop > 8


def update_occupancy():
    for track in tracks:
        matched_seat = min([seat for seat in seats if inside(track["point"], seat_region(seat["box"]))], key=lambda seat: np.linalg.norm(center(seat["box"]) - track["point"]), default=None)
        if matched_seat and track["seat"] is not matched_seat and track["seat"]:
            track["seat"].update({"occupied": False, "occupant": None, "away": 0})
        track["dwell"] = track["dwell"] + 1 if matched_seat and track["seat"] is matched_seat else 1 if matched_seat else 0
        track["seat"] = matched_seat
        if matched_seat and track["dwell"] >= 2 and sat_down(track, matched_seat):
            matched_seat.update({"occupied": True, "occupant": track["id"], "away": 0})
    for seat in seats:
        occupant = next((track for track in tracks if track["id"] == seat["occupant"]), None)
        seat["away"] = 0 if occupant and inside(occupant["point"], seat_region(seat["box"])) else seat["away"] + 1
        if seat["away"] > 8:
            seat.update({"occupied": False, "occupant": None, "away": 0})
    occupied = [seat for seat in seats if seat["occupied"]]
    seats[:] = [seat for seat in seats if seat["occupied"] or not any(inside(center(seat["box"]), taken["box"]) or overlap(seat["box"], taken["box"]) / max(area(seat["box"]), 1) > 0.35 for taken in occupied)]


while True:
    ok, image = camera.read()
    if not ok:
        break

    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    compensate_camera_motion(gray, image_width, image_height)
    detected = frame_segments(image)
    person_masks = [mask for label, mask, box in detected if label == 0]
    chair_masks = [mask for label, mask, box in detected if label == 56]
    person_boxes = merge_boxes(np.array([box for label, mask, box in detected if label == 0], dtype=np.float32))
    chair_boxes = filter_seats(split_benches(merge_boxes(np.array([box for label, mask, box in detected if label == 56], dtype=np.float32), 0.25, 0.45)))
    update_seats(chair_boxes)
    update_tracks(person_boxes)
    update_occupancy()
    head_areas = np.maximum((person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]) * 0.12, 1) if len(person_boxes) else np.array([0.0])
    density = float(len(person_boxes) * head_areas.mean() / (image_height * image_width))
    crowdedness_pct = min(100, round(100 * density / 0.155, 1))

    for mask in chair_masks:
        image[mask] = image[mask] * 0.55 + np.array([0, 0, 255]) * 0.45
    for mask in person_masks:
        image[mask] = image[mask] * 0.55 + np.array([255, 0, 0]) * 0.45
    visible_tracks = [track for track in tracks if track["missed"] < 3]
    for track in visible_tracks:
        x1, y1, x2, y2 = track["box"]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.circle(image, tuple(track["point"].astype(int)), 4, (255, 255, 0), -1)
        cv2.putText(image, f"Person {track['id']}", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for seat in seats:
        x1, y1, x2, y2 = seat["box"]
        color = (0, 255, 0) if seat["occupied"] else (0, 0, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, "Seat occupied" if seat["occupied"] else "Seat empty", (int(x1), max(25, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    occupied_count = sum(seat["occupied"] for seat in seats)
    cv2.putText(image, f"Crowdedness: {crowdedness_pct}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"People: {len(visible_tracks)}  Seats tracked: {len(seats)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Density: {density:.4f}  Avg head px: {head_areas.mean():.1f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Seats occupied: {occupied_count}/{len(seats)}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Bus Crowdedness Swin Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
