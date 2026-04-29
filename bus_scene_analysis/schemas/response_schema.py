FINAL_RESPONSE_SCHEMA = {
    "frame_id": "string",
    "timestamp": "string",
    "detections": "array",
    "yolo_metrics": "object",
    "llava_summary": "string|null",
    "alerts": [{"type": "string", "severity": "low|medium|high", "message": "string"}],
}


def alert(alert_type, severity, message):
    return {"type": alert_type, "severity": severity, "message": message}


def final_response(yolo_output, llava_summary, alerts):
    return {
        "frame_id": yolo_output["frame_id"],
        "timestamp": yolo_output["timestamp"],
        "detections": yolo_output["detections"],
        "yolo_metrics": yolo_output["metrics"],
        "llava_summary": llava_summary,
        "alerts": alerts,
    }

