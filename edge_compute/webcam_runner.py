import logging
import sys

import cv2

from bus_scene_analysis import config
from edge_compute.scene_analyzer import EdgeSceneAnalyzer
from edge_compute.video_runner import draw_analysis


def run():
    logging.basicConfig(level=config.LOG_LEVEL, format="%(message)s")
    analyzer = EdgeSceneAnalyzer(enable_llava=config.USE_LLAVA)
    camera_source = next((arg for arg in sys.argv[1:] if not arg.startswith("--")), "0")
    camera = cv2.VideoCapture(int(camera_source) if camera_source.isdigit() else camera_source)
    frame_number, summary = 0, ""
    while True:
        ok, image = camera.read()
        if not ok:
            break
        frame_number += 1
        fps = camera.get(cv2.CAP_PROP_FPS) or 30
        analysis = analyzer.analyze_frame(image, frame_id=f"webcam:{frame_number}", fps=fps, include_image=config.LLAVA_INCLUDE_IMAGE, track=frame_number > 1, run_backend=frame_number % max(1, int(fps * config.BACKEND_SAMPLE_SECONDS)) == 1)
        summary = draw_analysis(image, analysis, summary)
        cv2.imshow("Bus Scene Analysis", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
