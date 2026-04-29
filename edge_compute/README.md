# Edge compute

Local runtime code only:

- `yolo_detector.py`: YOLO26n people detection plus fine-tuned empty-seat detection.
- `vision_encoder.py`: SigLIP image feature-vector payload prep for LLaVA-side summaries.
- `scene_analyzer.py`: runs YOLO first, then calls the local server-side language model file.
- `video_runner.py`: processes raw videos from `data/` and writes annotated videos to `output/annotated/`.
- `webcam_runner.py`: live camera runner.

Commands:

```powershell
python -m edge_compute.video_runner
python -m edge_compute.webcam_runner
```
