# YOLO fine-tuning

This folder contains the empty-seat fine-tuning assets only:

- `train_yolo_seats.py`: retrains the fine-tuned empty-seat detector from YOLO26n.
- `seat_labeler.py`: tiny browser labeler that writes labels to `datasets/seat_labels/`.
- `datasets/`: derived YOLO datasets and extracted training frames.
- `runs/`: training outputs and the active fine-tuned seat weights.

The people detector now uses the base `yolo26n.pt` model. It was not the fine-tuned model, so no retraining was needed for that conversion.
