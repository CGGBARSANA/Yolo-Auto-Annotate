import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO  # Works for YOLOv5/8/11 if compatible
from pathlib import Path

# Configuration
IMAGE_DIR = r'C:\Users\BARSANA FAMILY\Documents\Projects\autotate\images'
LABEL_DIR = r'C:\Users\BARSANA FAMILY\Documents\Projects\autotate\labels'
MODEL_PATH = r'C:\Users\BARSANA FAMILY\Documents\Projects\autotate\models\yolo11n.pt'
ANNOTATED_DIR = r'C:\Users\BARSANA FAMILY\Documents\Projects\autotate\annotations'
CONF_THRESHOLD = 0.3

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Class names (edit if custom classes)
class_names = model.names

for img_file in tqdm(os.listdir(IMAGE_DIR)):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    results = model(img_path, conf=CONF_THRESHOLD)[0]

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Draw annotations and write .txt
    label_path = Path(LABEL_DIR) / (Path(img_file).stem + ".txt")
    with open(label_path, "w") as f:
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # YOLO format save
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[cls]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image
    out_path = os.path.join(ANNOTATED_DIR, img_file)
    cv2.imwrite(out_path, img)