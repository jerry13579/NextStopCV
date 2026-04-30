import os, cv2, torch
import torch.nn as nn
from glob import glob
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.models.siglip import SiglipVisionModel

VIDEO_GLOB = "data/IMG_*.mov"
OUTPUT_DIR = "output/custom_vlm_360p_reasoning_1080p"
CHECKPOINT = "custom_split_vlm_output/content/custom_split_vlm/checkpoints/stage3_vqa/checkpoint-1000/model.safetensors"
VISION_MODEL_ID = "google/siglip-base-patch16-224"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE, DTYPE, SAMPLE_SECONDS, MAX_NEW_TOKENS = "cuda:0", torch.bfloat16, 5, 56
PROMPT = "Look at the bus interior and answer in one qualitative sentence. Use this format: Occupancy feels light-to-moderate; passengers are mostly seated; seating appears available; center aisle feels open; boarding flow appears calm. Change only clearly wrong words. No counts."


class SplitSigLIPVision(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.embeddings, self.pi_layers, self.backend_layers, self.post_layernorm = base.embeddings, base.encoder.layers[:2], base.encoder.layers[2:], base.post_layernorm
        self.compressor, self.decompressor = nn.Linear(768, 256), nn.Linear(256, 768)
        weights = torch.zeros(196, 64)
        for i in range(64):
            start, end = i * 196 // 64, ((i + 1) * 196 + 63) // 64
            weights[start:end, i] = 1 / (end - start)
        self.register_buffer("pool_weights", weights)

    def encode_pi_side(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.pi_layers: x = layer(x, None)
        return self.compressor(x.transpose(1, 2).matmul(self.pool_weights.to(x.device, x.dtype)).transpose(1, 2))

    def decode_backend_side(self, compressed_tokens):
        x = self.decompressor(compressed_tokens)
        for layer in self.backend_layers: x = layer(x, None)
        return self.post_layernorm(x)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(768, 2048), nn.GELU(), nn.Linear(2048, 1536))

    def forward(self, x): return self.proj(x)


class CustomSplitVLM(nn.Module):
    def __init__(self):
        super().__init__()
        vision_cfg = AutoConfig.from_pretrained(VISION_MODEL_ID, local_files_only=True).vision_config
        self.processor, self.tokenizer = AutoProcessor.from_pretrained(VISION_MODEL_ID, local_files_only=True), AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self.vision_model, self.projector, self.llm = SplitSigLIPVision(SiglipVisionModel(vision_cfg)).to(DTYPE), Projector().to(DTYPE), Qwen2ForCausalLM(AutoConfig.from_pretrained(LLM_MODEL_ID)).to(DTYPE)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.load_checkpoint()
        self.to(DEVICE, DTYPE).eval()

    def load_checkpoint(self):
        with safe_open(CHECKPOINT, framework="pt", device="cpu") as f:
            self.vision_model.load_state_dict({k[13:]: f.get_tensor(k) for k in f.keys() if k.startswith("vision_model.")})
            self.projector.load_state_dict({k[10:]: f.get_tensor(k) for k in f.keys() if k.startswith("projector.")})
            self.llm.load_state_dict(self.qwen_state(f), strict=False)

    def qwen_state(self, f):
        state = {}
        for k in f.keys():
            if not k.startswith("llm.base_model.model.") or ".lora_" in k: continue
            name = k[21:].replace(".base_layer", "")
            if k.endswith(".base_layer.weight"):
                stem = k[:-18]
                state[name] = (f.get_tensor(k).float() + f.get_tensor(stem + ".lora_B.default.weight").float().matmul(f.get_tensor(stem + ".lora_A.default.weight").float()) * 2).to(DTYPE)
            else:
                state[name] = f.get_tensor(k)
        return state

    @torch.inference_mode()
    def summarize(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE, DTYPE)
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        visual = self.projector(self.vision_model.decode_backend_side(self.vision_model.encode_pi_side(pixels)))
        embeds = torch.cat([visual, self.llm.get_input_embeddings()(tokens.input_ids)], dim=1)
        mask = torch.cat([torch.ones((1, 64), dtype=tokens.attention_mask.dtype, device=DEVICE), tokens.attention_mask], dim=1)
        output = self.llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return clean(self.tokenizer.decode(output[0], skip_special_tokens=True))


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = CustomSplitVLM()
    for path in glob(VIDEO_GLOB): annotate_video(path, model)


def annotate_video(path, model):
    cap = cv2.VideoCapture(path)
    fps, summary, interval = cap.get(cv2.CAP_PROP_FPS) or 30, "", max(1, int((cap.get(cv2.CAP_PROP_FPS) or 30) * SAMPLE_SECONDS))
    source_w, source_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    infer_size, render_size = aspect_size(source_w, source_h, 360), aspect_size(source_w, source_h, 1080)
    out_path = f"{OUTPUT_DIR}/{os.path.splitext(os.path.basename(path))[0]}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, render_size)
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_no % interval == 0: summary = model.summarize(cv2.resize(frame, infer_size, interpolation=cv2.INTER_AREA))
        render = cv2.resize(frame, render_size, interpolation=cv2.INTER_CUBIC)
        draw_reasoning(render, summary)
        writer.write(render)
        frame_no += 1
        print(f"{os.path.basename(path)} frame {frame_no}", end="\r")
    cap.release(); writer.release()
    print(f"wrote {out_path}".ljust(90))


def aspect_size(w, h, target_h):
    return (round(w * target_h / h) // 2 * 2, target_h)


def draw_reasoning(image, text):
    h, w = image.shape[:2]
    scale, thick = h / 1080 * 0.62, max(1, round(h / 540))
    lines, gap = wrap_text("LLM reasoning: " + text, w - 80, scale, thick)[:5], round(h / 34)
    top = h - 24 - gap * len(lines) - 24
    cv2.rectangle(image, (24, top), (w - 24, h - 24), (0, 0, 0), -1)
    for i, line in enumerate(lines): cv2.putText(image, line, (44, top + round(h / 38) + i * gap), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def wrap_text(text, width, scale, thickness):
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= width: line = test
        else: lines.append(line); line = word
    return lines + [line]


def clean(text):
    for marker in ["assistant\n", "assistant", "Answer:"]:
        text = text.split(marker)[-1]
    text = " ".join(text.replace("\n", " ").strip().strip('"').split())
    return text.split(".")[0].strip() + "."


if __name__ == "__main__":
    run()
