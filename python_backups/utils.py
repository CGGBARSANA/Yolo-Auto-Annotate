import cv2
import numpy as np

def preprocess(image, img_size):
    h0, w0 = image.shape[:2]
    r = img_size / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    image = image.transpose((2, 0, 1))[::-1].copy() / 255.0  # BGR to RGB, HWC to CHW, normalize
    image = np.expand_dims(image.astype(np.float32), 0)
    return image, r, (dw, dh)

def postprocess(predictions, conf_thres, iou_thres, ratio, pad_x, pad_y, original_shape):
    boxes = []
    for pred in predictions:
        if pred[4] < conf_thres:
            continue
        x1, y1, x2, y2 = pred[:4]
        conf, cls = pred[4], int(pred[5])
        # Adjust back to original image size
        x1 = ((x1 - pad_x) / ratio)
        y1 = ((y1 - pad_y) / ratio)
        x2 = ((x2 - pad_x) / ratio)
        y2 = ((y2 - pad_y) / ratio)
        boxes.append([cls, x1, y1, x2, y2])
    return boxes

def save_yolo_annotation(detections, file_path, img_w, img_h):
    with open(file_path, 'w') as f:
        for cls, x1, y1, x2, y2 in detections:
            # Convert to YOLO format: class cx cy w h (normalized)
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")