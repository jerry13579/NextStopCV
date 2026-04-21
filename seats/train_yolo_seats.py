import os, random, shutil
from ultralytics import YOLO

classes = ["empty_seat"]
root = "data/yolo_empty_seats"
src_images = "data"
src_labels = "data/seat_labels"
runs = "data/runs/yolo26_empty_seats"
seed = 42
random.seed(seed)
image_names = sorted(f for f in os.listdir(src_images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
items = [(os.path.splitext(f)[0], f) for f in image_names]
split = max(1, round(len(items) * 0.2))
val = {stem for stem, _ in random.sample(items, split)}
for part in ("train", "val"):
    for kind in ("images", "labels"):
        path = f"{root}/{part}/{kind}"
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
for stem, image_name in items:
    part = "val" if stem in val else "train"
    shutil.copy2(f"{src_images}/{image_name}", f"{root}/{part}/images/{image_name}")
    label_name = f"{stem}.txt"
    label_path = f"{src_labels}/{label_name}"
    rows = [] if not os.path.exists(label_path) else [" ".join(["0", *line.split()[1:]]) for line in open(label_path) if line.strip() and int(line.split()[0]) == 0]
    open(f"{root}/{part}/labels/{label_name}", "w").write("\n".join(rows) + ("\n" if rows else ""))
yaml = f"path: {os.path.abspath(root)}\ntrain: train/images\nval: val/images\nnames:\n  0: empty_seat\n"
open(f"{root}/dataset.yaml", "w").write(yaml)
device = "cpu"
model = YOLO("yolo26n.pt")
model.train(data=f"{root}/dataset.yaml", project=runs, name="train_all_images", epochs=20, imgsz=640, batch=4, device=device, workers=0, pretrained=True)
