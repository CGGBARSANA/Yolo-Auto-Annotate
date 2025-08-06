import sys
import os
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QMouseEvent
from PyQt5.QtCore import Qt, QRect


class AnnotatableLabel(QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.setPixmap(pixmap)
        self.image = pixmap
        self.drawing_enabled = False
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.boxes = []
        self.setMouseTracking(True)

    def enable_drawing(self):
        self.drawing_enabled = True

    def clear_boxes(self):
        self.boxes = []
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.drawing_enabled:
            self.drawing = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_enabled and self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.drawing_enabled and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            self.boxes.append(rect)
            self.start_point = self.end_point = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() is None:
            return

        painter = QPainter(self)
        painter.setPen(QPen(Qt.green, 2))

        for rect in self.boxes:
            painter.drawRect(rect)

        if self.drawing_enabled and self.drawing and self.start_point and self.end_point:
            temp_rect = QRect(self.start_point, self.end_point).normalized()
            painter.drawRect(temp_rect)


class ImageAnnotator(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Annotator")
        self.image_path = image_path
        self.image = QPixmap(self.image_path)

        self.label = AnnotatableLabel(self.image)

        # === Buttons ===
        self.start_button = QPushButton("Start Annotation")
        self.retry_button = QPushButton("Retry")
        self.save_button = QPushButton("Save Annotations")

        self.start_button.clicked.connect(self.start_annotation)
        self.retry_button.clicked.connect(self.retry_annotation)
        self.save_button.clicked.connect(self.save_annotations)

        # === Layout ===
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.retry_button)
        btn_layout.addWidget(self.save_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.resize(self.image.width(), self.image.height() + 60)

    def start_annotation(self):
        self.label.enable_drawing()
        print("🟢 Annotation mode started.")

    def retry_annotation(self):
        self.label.clear_boxes()
        print("🔁 All annotations cleared.")

    def save_annotations(self):
        image_width = self.image.width()
        image_height = self.image.height()

        output_txt = "output.txt"
        output_json = "output.json"

        default_cls = 33
        default_conf = 0.95
        default_label = "Jonathan"

        # Save .txt (YOLO format)
        with open(output_txt, "w") as f:
            for rect in self.label.boxes:
                x_center = (rect.left() + rect.right()) / 2 / image_width
                y_center = (rect.top() + rect.bottom()) / 2 / image_height
                w = rect.width() / image_width
                h = rect.height() / image_height
                f.write(f"{default_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Save .json (custom format)
        detections = []
        box_labels = {}
        selected_boxes = []

        for idx, rect in enumerate(self.label.boxes):
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            detections.append({
                "cls": default_cls,
                "conf": default_conf,
                "box": [x1, y1, x2, y2],
                "area": rect.width() * rect.height(),
                "original_index": idx
            })
            selected_boxes.append(idx)
            box_labels[str(idx)] = default_label

        data = {
            "image_path": os.path.abspath(self.image_path),
            "detections": detections,
            "selected_boxes": selected_boxes,
            "box_labels": box_labels,
            "original_shape": [image_height, image_width],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "class_selection_mode": "both"
        }

        with open(output_json, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved to {output_txt} and {output_json}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    image_path, _ = QFileDialog.getOpenFileName(
        None, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
    )
    if image_path:
        window = ImageAnnotator(image_path)
        window.show()
        sys.exit(app.exec_())
