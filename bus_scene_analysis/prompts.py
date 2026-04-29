import json


def bus_scene_prompt(yolo_output):
    metrics_json = json.dumps(yolo_output["metrics"], indent=2)
    detections_json = json.dumps(yolo_output["detections"][:40], indent=2)
    return f"""You are analyzing a bus interior scene. Describe the scene in general for a transit operator. Use the YOLO metrics only as context and do not invent exact counts.

YOLO metrics:
{metrics_json}

YOLO detections sample:
{detections_json}

Tasks:
1. Describe the overall crowding and seat availability.
2. Mention any likely aisle, boarding, safety, or accessibility concerns.
3. Give one concise operator-facing summary without listing exact counts."""
