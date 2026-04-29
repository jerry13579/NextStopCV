from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="yolo_finetuning/datasets/yolo_empty_seats_all_current/dataset.yaml", epochs=100, imgsz=640, batch=8, device=0, workers=0, project="yolo_finetuning/runs/yolo26_empty_seats", name="train_all_current_labels", exist_ok=True)
