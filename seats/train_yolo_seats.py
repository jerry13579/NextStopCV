import os, random, shutil
from ultralytics import YOLO

root = "data/yolo_empty_seats_all_current"
src_images = "data"
src_labels = "data/seat_labels"
runs = os.path.abspath("runs/yolo26_empty_seats")
seed = 42
random.seed(seed)
items = []
for label_name in sorted(f for f in os.listdir(src_labels) if f.endswith(".txt")):
    stem = os.path.splitext(label_name)[0]
    image_name = next((f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if os.path.exists(f"{src_images}/{stem}{ext}")), None)
    if image_name: items.append((stem, image_name))
random.shuffle(items)
cut = max(1, int(len(items) * 0.8))
splits = {"train": items[:cut], "val": items[cut:] or items[:1]}
for part in ("train", "val"):
    for kind in ("images", "labels"):
        path = f"{root}/{part}/{kind}"
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
for part, rowset in splits.items():
    for stem, image_name in rowset:
        rows = ["0 " + " ".join(line.split()[1:]) for line in open(f"{src_labels}/{stem}.txt") if line.strip() and int(line.split()[0]) == 0]
        shutil.copy2(f"{src_images}/{image_name}", f"{root}/{part}/images/{image_name}")
        open(f"{root}/{part}/labels/{stem}.txt", "w").write("\n".join(rows) + ("\n" if rows else ""))
open(f"{root}/dataset.yaml", "w").write(f"path: {os.path.abspath(root)}\ntrain: train/images\nval: val/images\nnames:\n  0: empty_seat\n")
YOLO("yolo26n.pt").train(data=f"{root}/dataset.yaml", project=runs, name="train_all_current_labels", epochs=100, imgsz=640, batch=8, device=0, workers=0, pretrained=True)
