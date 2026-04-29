DETECTION_OUTPUT_SCHEMA = {
    "timestamp": "string",
    "frame_id": "string",
    "detections": [{"class": "string", "confidence": "number", "bbox": ["number", "number", "number", "number"]}],
    "metrics": {
        "passenger_count": "integer",
        "empty_seat_count": "integer|null",
        "occupied_seat_count": "integer|null",
        "standing_passenger_count": "integer|null",
        "crowding_level": "low|moderate|high",
    },
}


def detection_item(name, confidence, bbox, track_id=None):
    item = {"class": name, "confidence": round(float(confidence), 4), "bbox": [round(float(x), 2) for x in bbox]}
    if track_id is not None:
        item["track_id"] = int(track_id)
    return item


def detection_output(timestamp, frame_id, detections, metrics):
    return {"timestamp": timestamp, "frame_id": str(frame_id), "detections": detections, "metrics": metrics}

