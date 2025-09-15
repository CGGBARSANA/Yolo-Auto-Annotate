import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("models/yolo11n.pt")  # replace with your model path

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame, conf=0.70)

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()   # Bounding boxes
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Draw box and manage_label_btn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("YOLO Camera", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
