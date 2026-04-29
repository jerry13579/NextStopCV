import json
import logging

from bus_scene_analysis import config
from bus_scene_analysis.alerts import scene_alerts
from bus_scene_analysis.prompts import bus_scene_prompt
from bus_scene_analysis.schemas.response_schema import final_response
from edge_compute.vision_encoder import edge_vision_encoder
from edge_compute.yolo_detector import BusYoloDetector
from server_compute.language_model import LlavaOneVisionLanguageModel

logger = logging.getLogger(__name__)


class EdgeSceneAnalyzer:
    def __init__(self, yolo_detector=None, vision_encoder=None, enable_llava=config.USE_LLAVA):
        self.yolo = yolo_detector or BusYoloDetector()
        self.vision_encoder = vision_encoder
        self.language_model = None
        self.enable_llava = enable_llava

    def reset(self):
        self.yolo.reset()

    def analyze_frame(self, image, frame_id="0", timestamp=None, fps=30, include_image=True, track=True, run_backend=True):
        yolo_output = self.yolo.analyze_frame(image, frame_id, timestamp, fps, track)
        llava_summary = None
        if self.enable_llava and run_backend:
            prompt = bus_scene_prompt(yolo_output)
            vision_payload = self.encode_vision(image) if include_image else None
            logger.debug("Edge LLaVA request: %s", json.dumps({"frame_id": yolo_output["frame_id"], "metrics": yolo_output["metrics"], "vision_payload_kind": vision_payload["kind"] if vision_payload else None}))
            try:
                llava_summary = self.local_summary(prompt, vision_payload, yolo_output)
            except Exception as exc:
                logger.exception("Server LLaVA failed; returning YOLO-only metrics: %s", exc)
        response = final_response(yolo_output, llava_summary, scene_alerts(yolo_output["metrics"], llava_summary))
        logger.debug("Edge final response: %s", json.dumps(response))
        return response

    def encode_vision(self, image):
        self.vision_encoder = self.vision_encoder or edge_vision_encoder()
        return self.vision_encoder.encode(image)

    def local_summary(self, prompt, vision_payload, yolo_output):
        self.language_model = self.language_model or LlavaOneVisionLanguageModel()
        return self.language_model.summarize(prompt, vision_payload, yolo_output)
