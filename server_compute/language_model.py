from bus_scene_analysis import config


class LlavaOneVisionLanguageModel:
    def summarize(self, prompt, vision_payload=None, yolo_output=None):
        import base64
        import numpy as np
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        if not hasattr(self, "model"):
            self.processor = AutoProcessor.from_pretrained(config.LLAVA_MODEL_ID)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(config.LLAVA_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(config.LLAVA_SERVER_DEVICE).eval()
        metrics = yolo_output["metrics"] if yolo_output else {}
        features = self.decode_features(vision_payload, base64, np, torch)
        seat_status = "no open seating detected" if int(metrics.get("empty_seat_count") or 0) == 0 else "some open seating appears available"
        crowding = metrics.get("crowding_level")
        text = f"Describe this bus interior status in one natural operator-facing sentence under 30 words. Do not list exact counts or describe lighting, colors, or unseen details. Required facts: crowding is {crowding}; {seat_status}. Mention aisle or boarding concerns only if appropriate."
        messages = [{"role": "user", "content": ([{"type": "image"}] if features is not None else []) + [{"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], return_tensors="pt").to(config.LLAVA_SERVER_DEVICE)
        inputs = self.with_features(inputs, features, torch) if features is not None else inputs
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=config.LLAVA_MAX_NEW_TOKENS, do_sample=False)
        start = 0 if isinstance(inputs, dict) else inputs.input_ids.shape[-1]
        return self.processor.batch_decode(output[:, start:], skip_special_tokens=True)[0].split("assistant")[-1].strip().strip('"').replace("\n", " ")

    def decode_features(self, vision_payload, base64, np, torch):
        if not vision_payload or vision_payload.get("kind") != "llava_onevision_vision_features":
            return None
        return torch.from_numpy(np.frombuffer(base64.b64decode(vision_payload["data"]), dtype=np.float16).reshape(vision_payload["shape"]))

    def with_features(self, inputs, features, torch):
        image_token_id = self.model.config.image_token_id if hasattr(self.model.config, "image_token_id") else self.model.config.image_token_index
        image_idx = int((inputs.input_ids[0] == image_token_id).nonzero()[0, 0])
        embeds = self.model.get_input_embeddings()(inputs.input_ids)
        projected = self.model.model.multi_modal_projector(features.to(config.LLAVA_SERVER_DEVICE, dtype=embeds.dtype))[0]
        inputs_embeds = torch.cat([embeds[:, :image_idx], projected[None], embeds[:, image_idx + 1:]], dim=1)
        return {"inputs_embeds": inputs_embeds, "attention_mask": torch.ones(inputs_embeds.shape[:2], dtype=inputs.attention_mask.dtype, device=inputs_embeds.device)}
