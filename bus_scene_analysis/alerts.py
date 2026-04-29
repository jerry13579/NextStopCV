from bus_scene_analysis.schemas.response_schema import alert


def scene_alerts(metrics, llava_summary=None):
    alerts = []
    if metrics["crowding_level"] == "high":
        alerts.append(alert("crowding", "high", f"Bus appears highly crowded at {metrics['crowdedness_pct']}% YOLO crowding."))
    elif metrics["crowding_level"] == "moderate":
        alerts.append(alert("crowding", "medium", "Bus appears moderately crowded; monitor aisle space and seat availability."))
    if metrics["empty_seat_count"] == 0:
        alerts.append(alert("seating", "medium", "YOLO found no empty seats in this frame window."))
    if metrics["door_area_congestion"] == "high":
        alerts.append(alert("door_congestion", "medium", "YOLO indicates congestion near the door area."))
    return alerts

