# NextStop Bus Scene Analysis

Hybrid bus interior analysis:

- `data/`: raw videos and raw source images only.
- `edge_compute/`: YOLO runtime, rolling metrics, SigLIP image feature payload prep, video/webcam runners.
- `server_compute/`: LLaVA-OneVision language-model summary code only.
- `bus_scene_analysis/`: shared config, prompts, schemas, and alert helpers.
- `yolo_finetuning/`: empty-seat training datasets, labeler, train script, and training runs.
- `output/annotated/`: rendered analysis videos.

Run the sample videos:

```powershell
python -m edge_compute.video_runner
```

Training is separate:

```powershell
python yolo_finetuning/train_yolo_seats.py
```

Current models:

- People detector: `yolo26n.pt`
- Empty-seat detector: `yolo_finetuning/runs/yolo26_empty_seats/train_all_current_labels/weights/best.pt`

Edge-to-server vision payloads are SigLIP feature vectors sampled every 5 seconds.
