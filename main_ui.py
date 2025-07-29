import os
import cv2
import sys
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QListWidget, QLineEdit, QMessageBox, QComboBox, QSpinBox,
    QGroupBox, QGridLayout, QScrollArea, QFrame, QSplitter, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from ultralytics import YOLO


class ImageThumbnail(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, image_path, index, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.index = index
        self.is_annotated = False
        self.setFixedSize(150, 150)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: white;
            }
            QLabel:hover {
                border-color: #0078d4;
            }
        """)
        self.load_thumbnail()

    def load_thumbnail(self):
        img = cv2.imread(self.image_path)
        if img is not None:
            # Resize image for thumbnail
            h, w = img.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w, new_h = 140, int(140 / aspect)
            else:
                new_w, new_h = int(140 * aspect), 140

            img = cv2.resize(img, (new_w, new_h))
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap)

        # Add filename as tooltip and bottom text
        filename = os.path.basename(self.image_path)
        self.setToolTip(filename)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)

    def set_current(self, is_current):
        if is_current:
            self.setStyleSheet("""
                QLabel {
                    border: 3px solid #0078d4;
                    border-radius: 5px;
                    background-color: #e6f3ff;
                }
            """)
        else:
            color = "#00ff04" if self.is_annotated else "#ccc"
            self.setStyleSheet(f"""
                QLabel {{
                    border: 2px solid {color};
                    border-radius: 5px;
                    background-color: white;
                }}
                QLabel:hover {{
                    border-color: #0078d4;
                }}
            """)

    def set_annotated(self, is_annotated):
        self.is_annotated = is_annotated
        if not hasattr(self, '_is_current') or not self._is_current:
            self.set_current(False)


class Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11 Auto Annotate GUI - Enhanced with Grid View")
        self.setGeometry(100, 100, 1400, 900)

        # Label assignment controls
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setPlaceholderText("Select or enter label")

        self.assign_label_btn = QPushButton("Assign Label to Selected")
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")

        self.load_btn = QPushButton("Load Images")
        self.next_btn = QPushButton("Next Image")
        self.prev_btn = QPushButton("Previous Image")
        self.save_btn = QPushButton("Save Annotation")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        # Status labels
        self.status_label = QLabel("Load images to start")
        self.selection_count_label = QLabel("Selected: 0")

        self.image_paths = []
        self.current_index = 0
        self.detections = []
        self.selected_boxes = set()
        self.class_names = []
        self.box_labels = {}  # Store custom labels for boxes
        self.thumbnails = []  # Store thumbnail widgets
        self.annotated_images = set()  # Track which images have been annotated
        self.has_unsaved_changes = False  # Track if current annotation has unsaved changes

        self.model = YOLO(r"C:\Users\Nyuu Sutairu IT Dep\Desktop\cleaned\autotate\models\large.pt")
        self.label_dir = r"C:\Users\Nyuu Sutairu IT Dep\Desktop\cleaned\autotate\labels"
        self.annotation_save_dir = r"C:\Users\Nyuu Sutairu IT Dep\Desktop\cleaned\autotate\saved_annotations"
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.annotation_save_dir, exist_ok=True)

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        main_layout = QHBoxLayout()

        # Create tab widget for grid and detail views
        self.tab_widget = QTabWidget()

        # Tab 1: Grid View
        grid_tab = QWidget()
        grid_layout = QVBoxLayout()

        # Grid controls
        grid_controls = QHBoxLayout()
        self.load_btn = QPushButton("Load Images")
        grid_controls.addWidget(self.load_btn)
        grid_controls.addStretch()
        grid_layout.addLayout(grid_controls)

        # Scrollable grid area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        scroll_area.setWidget(self.grid_widget)

        grid_layout.addWidget(scroll_area)
        grid_tab.setLayout(grid_layout)
        self.tab_widget.addTab(grid_tab, "Grid View")

        # Tab 2: Detail View
        detail_tab = QWidget()
        detail_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(300)
        left_widget.setLayout(left_panel)

        # Image info section
        info_group = QGroupBox("Image Info")
        info_layout = QVBoxLayout()
        self.status_label = QLabel("Load images to start")
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        left_panel.addWidget(info_group)

        # Label assignment section
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setPlaceholderText("Select or enter label")
        self.assign_label_btn = QPushButton("Assign Label to Selected")

        label_group = QGroupBox("Label Assignment")
        label_layout = QVBoxLayout()
        label_layout.addWidget(QLabel("Assign label to selected boxes:"))
        label_layout.addWidget(self.label_combo)
        label_layout.addWidget(self.assign_label_btn)
        label_group.setLayout(label_layout)
        left_panel.addWidget(label_group)

        # Selection controls
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.selection_count_label = QLabel("Selected: 0")

        selection_group = QGroupBox("Selection Controls")
        selection_layout = QVBoxLayout()
        selection_layout.addWidget(self.selection_count_label)
        selection_layout.addWidget(self.select_all_btn)
        selection_layout.addWidget(self.deselect_all_btn)
        selection_group.setLayout(selection_layout)
        left_panel.addWidget(selection_group)

        # Navigation controls
        self.next_btn = QPushButton("Next Image")
        self.prev_btn = QPushButton("Previous Image")
        self.save_btn = QPushButton("Save Annotation")
        self.auto_save_btn = QPushButton("Auto Save: ON")
        self.auto_save_btn.setCheckable(True)
        self.auto_save_btn.setChecked(True)
        self.auto_save_enabled = True

        nav_group = QGroupBox("Navigation & Save")
        nav_layout = QVBoxLayout()
        nav_btn_layout = QHBoxLayout()
        nav_btn_layout.addWidget(self.prev_btn)
        nav_btn_layout.addWidget(self.next_btn)
        nav_layout.addLayout(nav_btn_layout)
        nav_layout.addWidget(self.save_btn)
        nav_layout.addWidget(self.auto_save_btn)
        nav_group.setLayout(nav_layout)
        left_panel.addWidget(nav_group)

        # Session management
        session_group = QGroupBox("Session Management")
        session_layout = QVBoxLayout()

        self.clear_session_btn = QPushButton("Clear All Annotations")
        self.export_labels_btn = QPushButton("Export Selected Labels")

        session_layout.addWidget(self.clear_session_btn)
        session_layout.addWidget(self.export_labels_btn)
        session_group.setLayout(session_layout)
        left_panel.addWidget(session_group)

        left_panel.addStretch()

        # Right panel for image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        detail_layout.addWidget(left_widget)
        detail_layout.addLayout(right_layout)
        detail_tab.setLayout(detail_layout)
        self.tab_widget.addTab(detail_tab, "Detail View")

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def connect_signals(self):
        self.load_btn.clicked.connect(self.load_images)
        self.next_btn.clicked.connect(self.show_next_image)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.assign_label_btn.clicked.connect(self.assign_label_to_selected)
        self.select_all_btn.clicked.connect(self.select_all_boxes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_boxes)
        self.auto_save_btn.clicked.connect(self.toggle_auto_save)
        self.clear_session_btn.clicked.connect(self.clear_all_annotations)
        self.export_labels_btn.clicked.connect(self.export_selected_labels)
        self.image_label.mousePressEvent = self.handle_click

    def load_images(self):
        self.class_names = self.model.names  # Automatically from model

        # Populate combo box with model class names
        self.label_combo.clear()
        for class_name in self.class_names.values():
            self.label_combo.addItem(class_name)

        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.jpg *.png *.jpeg)")
        if files:
            self.image_paths = files
            self.current_index = 0
            self.annotated_images.clear()
            self.has_unsaved_changes = False
            self.create_grid_thumbnails()
            self.show_current_image()
            self.status_label.setText(f"Loaded {len(files)} images")
            # Switch to detail view after loading
            self.tab_widget.setCurrentIndex(1)

    def create_grid_thumbnails(self):
        # Clear existing thumbnails
        for thumbnail in self.thumbnails:
            thumbnail.deleteLater()
        self.thumbnails.clear()

        # Clear grid layout
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        # Create new thumbnails
        cols = 5  # Number of columns in grid
        for i, img_path in enumerate(self.image_paths):
            thumbnail = ImageThumbnail(img_path, i)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(thumbnail, row, col)
            self.thumbnails.append(thumbnail)

        # Update current thumbnail
        if self.thumbnails:
            self.thumbnails[self.current_index].set_current(True)

    def on_thumbnail_clicked(self, index):
        # Auto-save current annotation before switching if enabled
        if self.auto_save_enabled and self.image_paths and self.detections and hasattr(self, 'current_index'):
            self.auto_save_current_annotation()

        self.current_index = index
        self.show_current_image()
        # Switch to detail view
        self.tab_widget.setCurrentIndex(1)
        # Update thumbnail states
        self.update_thumbnail_states()

    def show_current_image(self):
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return

        img_path = self.image_paths[self.current_index]
        self.current_image = cv2.imread(img_path)
        self.original_shape = self.current_image.shape[:2]

        # Try to load previous annotation first
        if self.load_previous_annotation():
            # Previous annotation loaded successfully
            self.has_unsaved_changes = False
            print(f"Loaded previous annotation for {os.path.basename(img_path)}")
        else:
            # No previous annotation, run YOLO detection
            result = self.model(img_path, conf=0.3)[0]

            self.detections = []
            self.box_labels = {}

            for i, box in enumerate(result.boxes):
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)  # Calculate area for sorting
                self.detections.append({
                    "cls": cls,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "area": area,
                    "original_index": i
                })

            # Sort detections by area (largest first, so smallest will be drawn last and be on top)
            self.detections.sort(key=lambda x: x["area"], reverse=True)

            # Auto-select all boxes
            self.selected_boxes = set(range(len(self.detections)))
            self.has_unsaved_changes = True

        self.display_image()
        self.update_status()

    def show_next_image(self):
        # Auto-save current annotation before moving if enabled
        if self.auto_save_enabled and self.image_paths and self.detections:
            self.auto_save_current_annotation()

        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_thumbnail_states()
        else:
            QMessageBox.information(self, "Info", "This is the last image.")

    def show_prev_image(self):
        # Auto-save current annotation before moving if enabled
        if self.auto_save_enabled and self.image_paths and self.detections:
            self.auto_save_current_annotation()

        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.update_thumbnail_states()
        else:
            QMessageBox.information(self, "Info", "This is the first image.")

    def update_thumbnail_states(self):
        for i, thumbnail in enumerate(self.thumbnails):
            thumbnail.set_current(i == self.current_index)
            # Check if annotation file exists
            annotation_exists = self.has_saved_annotation(i)
            thumbnail.set_annotated(annotation_exists)

    def has_saved_annotation(self, image_index):
        """Check if an image has a saved annotation"""
        annotation_file = self.get_annotation_filename(image_index)
        return os.path.exists(annotation_file)

    def display_image(self):
        if not hasattr(self, 'current_image'):
            return

        img = self.current_image.copy()

        # Draw boxes in order (largest first, so smallest appears on top)
        for i, det in enumerate(self.detections):
            x1, y1, x2, y2 = det["box"]
            cls = det["cls"]

            # Choose color based on selection
            if i in self.selected_boxes:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
            else:
                color = (0, 0, 255)  # Red for unselected
                thickness = 2

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Display label (custom or original class name)
            if i in self.box_labels:
                label_text = self.box_labels[i]
                # if i in self.box_labels:
                # custom_label = self.box_labels[i]
                # cls = None
                self.label_combo.setCurrentText(label_text)
                for idx, name in self.class_names.items():
                    if name == label_text:
                        det["cls"] = idx
                        break
            else:
                label_text = f"{self.class_names[cls]}"

            # Add confidence score
            label_text += f" ({det['conf']:.2f})"

            cv2.putText(img, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image)

        # Scale image to fit label while maintaining aspect ratio
        scaled_pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pix)

    def handle_click(self, event):
        if not self.detections:
            return

        x = event.pos().x()
        y = event.pos().y()
        label_w, label_h = self.image_label.width(), self.image_label.height()
        img_h, img_w = self.original_shape

        # Calculate resize ratio
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return

        scaled_w = pixmap.width()
        scaled_h = pixmap.height()
        x_offset = (label_w - scaled_w) / 2
        y_offset = (label_h - scaled_h) / 2

        # Convert click to image coordinates
        if x < x_offset or y < y_offset or x > x_offset + scaled_w or y > y_offset + scaled_h:
            return  # Click outside image

        img_x = int((x - x_offset) * img_w / scaled_w)
        img_y = int((y - y_offset) * img_h / scaled_h)

        # Check boxes in reverse order (smallest first, since they're drawn on top)
        for i in reversed(range(len(self.detections))):
            det = self.detections[i]
            x1, y1, x2, y2 = det["box"]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                if i in self.selected_boxes:
                    self.selected_boxes.remove(i)
                else:
                    self.selected_boxes.add(i)
                self.has_unsaved_changes = True
                self.display_image()
                self.update_status()
                break

    def assign_label_to_selected(self):
        if not self.selected_boxes:
            QMessageBox.warning(self, "Warning", "No boxes selected.")
            return

        label_text = self.label_combo.currentText().strip()
        if not label_text:
            QMessageBox.warning(self, "Warning", "Please enter or select a label.")
            return

        # Assign label to all selected boxes
        for i in self.selected_boxes:
            self.box_labels[i] = label_text

        # Add to combo box if it's a new label
        if self.label_combo.findText(label_text) == -1:
            self.label_combo.addItem(label_text)

        self.has_unsaved_changes = True
        self.display_image()

        self.save_annotations()
        # QMessageBox.information(self, "Success", f"Assigned label '{label_text}' to {len(self.selected_boxes)} boxes.")

    def select_all_boxes(self):
        self.selected_boxes = set(range(len(self.detections)))
        self.has_unsaved_changes = True
        self.display_image()
        self.update_status()

    def deselect_all_boxes(self):
        self.selected_boxes.clear()
        self.has_unsaved_changes = True
        self.display_image()
        self.update_status()

    def update_status(self):
        if self.image_paths:
            status_text = f"Image {self.current_index + 1} of {len(self.image_paths)}"
            if self.has_unsaved_changes:
                status_text += " (Modified)"
            elif self.has_saved_annotation(self.current_index):
                status_text += " (Saved)"
            self.status_label.setText(status_text)
        self.selection_count_label.setText(f"Selected: {len(self.selected_boxes)} / {len(self.detections)}")

    def save_annotations(self):
        if not self.detections or not self.selected_boxes:
            QMessageBox.warning(self, "Warning", "No boxes selected to save.")
            return

        # Save both YOLO format and annotation JSON
        self.save_yolo_format()
        self.save_current_annotation()
        self.has_unsaved_changes = False

        # Mark image as annotated and update thumbnails
        self.annotated_images.add(self.current_index)
        self.update_thumbnail_states()
        self.update_status()

        # img_name = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
        # QMessageBox.information(self, "Saved",
        #                         f"Annotation saved for {img_name} with {len(self.selected_boxes)} boxes.")

    def save_yolo_format(self):
        """Save annotations in YOLO format"""
        img_path = self.image_paths[self.current_index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        h, w = self.original_shape

        label_file = os.path.join(self.label_dir, f"{img_name}.txt")
        with open(label_file, "w") as f:
            for i in self.selected_boxes:
                det = self.detections[i]
                x1, y1, x2, y2 = det["box"]
                print(i)

                # Determine class index
                if i in self.box_labels:
                    custom_label = self.box_labels[i]
                    cls = None
                    for idx, name in self.class_names.items():
                        if name == custom_label:
                            cls = idx
                            break
                    print("LABEL in IF", cls, "|", self.box_labels, "|", det, "|")
                else:
                    cls = det["cls"]
                    print("LABEL in Else", cls)

                # Convert to YOLO format (normalized)
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def get_annotation_filename(self, image_index=None):
        """Get the annotation filename for current or specified image"""
        if image_index is None:
            image_index = self.current_index
        img_path = self.image_paths[image_index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.annotation_save_dir, f"{img_name}_annotation.json")

    def save_current_annotation(self):
        """Save current annotation state to JSON file"""
        if not self.image_paths:
            return

        annotation_data = {
            "image_path": self.image_paths[self.current_index],
            "detections": self.detections,
            "selected_boxes": list(self.selected_boxes),
            "box_labels": self.box_labels,
            "original_shape": self.original_shape,
            "timestamp": str(np.datetime64('now'))
        }

        annotation_file = self.get_annotation_filename()
        try:
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving annotation: {e}")
            return False

    def auto_save_current_annotation(self):
        """Auto-save current annotation if there are unsaved changes"""
        if self.has_unsaved_changes and self.detections:
            if self.save_current_annotation():
                print(f"Auto-saved annotation for image {self.current_index + 1}")
                self.save_annotations()

    def load_previous_annotation(self):
        """Load previous annotation if exists"""
        annotation_file = self.get_annotation_filename()

        if not os.path.exists(annotation_file):
            return False

        try:
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)

            # Verify the image path matches
            if annotation_data["image_path"] != self.image_paths[self.current_index]:
                return False

            self.detections = annotation_data["detections"]
            self.selected_boxes = set(annotation_data["selected_boxes"])
            self.box_labels = {int(k): v for k, v in annotation_data["box_labels"].items()}

            return True

        except Exception as e:
            print(f"Error loading annotation: {e}")
            return False

    def toggle_auto_save(self):
        """Toggle auto-save functionality"""
        self.auto_save_enabled = self.auto_save_btn.isChecked()
        if self.auto_save_enabled:
            self.auto_save_btn.setText("Auto Save: ON")
            # Auto-save current state if there are unsaved changes
            if self.has_unsaved_changes:
                self.auto_save_current_annotation()
        else:
            self.auto_save_btn.setText("Auto Save: OFF")

    def clear_all_annotations(self):
        """Clear all saved annotations"""
        reply = QMessageBox.question(self, 'Clear Annotations',
                                     'Are you sure you want to clear all saved annotations?\nThis cannot be undone.',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                # Remove all annotation files
                for filename in os.listdir(self.annotation_save_dir):
                    if filename.endswith('_annotation.json'):
                        os.remove(os.path.join(self.annotation_save_dir, filename))

                # Clear in-memory tracking
                self.annotated_images.clear()
                self.update_thumbnail_states()

                QMessageBox.information(self, "Cleared", "All annotations have been cleared.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error clearing annotations: {e}")

    def export_selected_labels(self):
        """Export only the final labels (YOLO format) for selected annotated images"""
        annotated_count = len([i for i in range(len(self.image_paths)) if self.has_saved_annotation(i)])

        if annotated_count == 0:
            QMessageBox.warning(self, "Warning", "No annotated images to export.")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return

        try:
            exported_count = 0
            for i in range(len(self.image_paths)):
                if self.has_saved_annotation(i):
                    # Load annotation and save as YOLO format
                    old_index = self.current_index
                    self.current_index = i

                    if self.load_previous_annotation():
                        img_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                        label_file = os.path.join(export_dir, f"{img_name}.txt")

                        # Get image dimensions
                        img = cv2.imread(self.image_paths[i])
                        h, w = img.shape[:2]

                        with open(label_file, "w") as f:
                            for box_idx in self.selected_boxes:
                                det = self.detections[box_idx]
                                x1, y1, x2, y2 = det["box"]

                                # Use custom label if assigned, otherwise use original class
                                cls = det["cls"]

                                # Convert to YOLO format (normalized)
                                cx = (x1 + x2) / 2 / w
                                cy = (y1 + y2) / 2 / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                        exported_count += 1
                    self.current_index = old_index
            QMessageBox.information(self, "Export Complete",
                                    f"Exported {exported_count} annotation files to:\n{export_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error during export: {e}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Annotator()
    window.show()
    sys.exit(app.exec_())