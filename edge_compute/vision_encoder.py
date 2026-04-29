import base64
import cv2
from PIL import Image

from bus_scene_analysis import config


class LlavaOneVisionEdgeEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(config.LLAVA_VISION_ENCODER_MODEL_ID)
        self.vision_model = AutoModel.from_pretrained(config.LLAVA_VISION_ENCODER_MODEL_ID, torch_dtype=torch.float32).to(config.LLAVA_ENCODER_DEVICE).eval()

    def encode(self, image):
        inputs = self.processor(images=self.pil_image(image), return_tensors="pt").to(config.LLAVA_ENCODER_DEVICE)
        with self.torch.no_grad():
            features = self.vision_model(**inputs).last_hidden_state.detach().cpu().to(self.torch.float16).numpy()
        return {"kind": "llava_onevision_vision_features", "encoder": config.LLAVA_VISION_ENCODER_MODEL_ID, "shape": list(features.shape), "dtype": "float16", "data": base64.b64encode(features.tobytes()).decode("ascii")}

    def pil_image(self, image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if hasattr(image, "shape") else Image.open(image).convert("RGB")


def edge_vision_encoder():
    return LlavaOneVisionEdgeEncoder()
