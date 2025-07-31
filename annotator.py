import os
import cv2
import json
import shutil
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QComboBox,
    QGroupBox, QGridLayout, QScrollArea, QTabWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from image_thumbnail import ImageThumbnail
from settings_dialog import SettingsDialog
from settings_manager import SettingsManager


class Annotator(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Initialize paths as None - will be set by settings
        self.model = None
        self.label_dir = None
        self.annotation_save_dir = None

        # Initialize other attributes
        self.image_paths = []
        self.current_index = 0
        self.detections = []
        self.selected_boxes = set()
        self.class_names = []
        self.box_labels = {}
        self.thumbnails = []
        self.annotated_images = set()
        self.has_unsaved_changes = False

        # Load settings or show settings dialog
        if not self.load_settings():
            self.show_settings_dialog()
        else:
            self.initialize_ui()

    def load_settings(self):
        """Load settings and initialize model"""
        if not self.settings_manager.settings_exist():
            return False

        settings = self.settings_manager.load_settings()

        try:
            # Initialize model
            self.model = YOLO(settings['model_path'])
            self.label_dir = settings['label_dir']
            self.annotation_save_dir = settings['annotation_save_dir']

            # Ensure directories exist
            os.makedirs(self.label_dir, exist_ok=True)
            os.makedirs(self.annotation_save_dir, exist_ok=True)

            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading settings: {e}")
            return False

    def show_settings_dialog(self):
        """Show settings configuration dialog"""
        current_settings = self.settings_manager.load_settings()
        self.settings_dialog = SettingsDialog(current_settings)
        self.settings_dialog.settings_saved.connect(self.on_settings_saved)
        self.settings_dialog.show()

    def on_settings_saved(self, settings):
        """Handle settings saved event"""
        if self.settings_manager.save_settings(settings):
            # Load the new settings
            if self.load_settings():
                self.initialize_ui()
                QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to apply new settings.")
        else:
            QMessageBox.critical(self, "Error", "Failed to save settings.")

    def initialize_ui(self):
        """Initialize the main UI after settings are loaded"""
        self.setWindowTitle("YOLOv11 Auto Annotate GUI - Enhanced with Grid View")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_ui()
        self.connect_signals()
        self.show()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Add settings button at the top
        settings_layout = QHBoxLayout()
        self.settings_btn = QPushButton("Settings")


        settings_layout.addWidget(self.settings_btn)
        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)

        # Create tab widget for grid and detail views
        self.tab_widget = QTabWidget()

        # Tab 1: Grid View
        grid_tab = QWidget()
        grid_layout = QVBoxLayout()

        # Grid controls

        grid_controls = QHBoxLayout()
        self.add_images_btn = QPushButton("Add Images")
        grid_controls.addWidget(self.add_images_btn)
        # self.load_btn = QPushButton("Load Images")
        # grid_controls.addWidget(self.load_btn)
        grid_controls.addStretch()
        grid_layout.addLayout(grid_controls)
        # In setup_ui method, in the grid_controls section:

        # In connect_signals method:

        self.remove_all_btn = QPushButton("Remove All Images")
        grid_controls.addWidget(self.remove_all_btn)
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; font-weight: bold;")
        grid_controls.addWidget(separator)

        # Thumbnail selection controls
        self.thumbnail_selection_label = QLabel("Selected: 0 / 0")
        grid_controls.addWidget(self.thumbnail_selection_label)

        self.select_all_thumbnails_btn = QPushButton("Select All")
        grid_controls.addWidget(self.select_all_thumbnails_btn)

        self.deselect_all_thumbnails_btn = QPushButton("Deselect All")
        grid_controls.addWidget(self.deselect_all_thumbnails_btn)

        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("QPushButton { background-color: #ffcccc; }")  # Light red background
        grid_controls.addWidget(self.remove_selected_btn)

        grid_controls.addStretch()
        grid_layout.addLayout(grid_controls)
        # In connect_signals method, add:

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
        # self.export_labels_btn = QPushButton("Export Selected Labels")
        self.export_annotations_btn = QPushButton("Export Annotations")

        session_layout.addWidget(self.clear_session_btn)
        # session_layout.addWidget(self.export_labels_btn)
        session_layout.addWidget(self.export_annotations_btn)
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
        # self.load_btn.clicked.connect(self.load_images)
        self.select_all_thumbnails_btn.clicked.connect(self.select_all_thumbnails)
        self.deselect_all_thumbnails_btn.clicked.connect(self.deselect_all_thumbnails)
        self.remove_selected_btn.clicked.connect(self.remove_selected_images)

        self.settings_btn.clicked.connect(self.show_settings_dialog)
        self.add_images_btn.clicked.connect(self.add_images)
        self.remove_all_btn.clicked.connect(self.remove_all_images)
        self.next_btn.clicked.connect(self.show_next_image)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.assign_label_btn.clicked.connect(self.assign_label_to_selected)
        self.select_all_btn.clicked.connect(self.select_all_boxes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_boxes)
        self.auto_save_btn.clicked.connect(self.toggle_auto_save)
        self.clear_session_btn.clicked.connect(self.clear_all_annotations)
        # self.export_labels_btn.clicked.connect(self.export_selected_labels)
        self.export_annotations_btn.clicked.connect(self.export_annotations)
        self.image_label.mousePressEvent = self.handle_click

    # def load_images(self):
    #     if not self.model:
    #         QMessageBox.warning(self, "Warning", "Please configure settings first.")
    #         self.show_settings_dialog()
    #         return
    #
    #     self.class_names = self.model.names
    #
    #     # Populate combo box with model class names
    #     self.label_combo.clear()
    #     for class_name in self.class_names.values():
    #         self.label_combo.addItem(class_name)
    #
    #     files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.jpg *.png *.jpeg)")
    #     if files:
    #         self.image_paths = files
    #         self.current_index = 0
    #         self.annotated_images.clear()
    #         self.has_unsaved_changes = False
    #         self.create_grid_thumbnails()
    #         self.show_current_image()
    #         self.status_label.setText(f"Loaded {len(files)} images")
    #         # Switch to detail view after loading
    #         self.tab_widget.setCurrentIndex(1)

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

            # Connect selection changed signal
            thumbnail.selection_changed.connect(self.update_thumbnail_selection_ui)

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(thumbnail, row, col)
            self.thumbnails.append(thumbnail)

        # Update current thumbnail
        if self.thumbnails:
            self.thumbnails[self.current_index].set_current(True)

        # Update selection UI
        self.update_thumbnail_selection_ui()
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
        if not self.image_paths or self.current_index >= len(self.image_paths) or not self.model:
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
        if self.auto_save_enabled:
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

    def remove_all_images(self):
        """Remove all loaded images from the thumbnail grid and reset the application state"""
        if not self.image_paths:
            QMessageBox.information(self, "Info", "No images are currently loaded.")
            return

        # Ask for confirmation
        reply = QMessageBox.question(self, 'Remove All Images',
                                     'Are you sure you want to remove all loaded images?\n'
                                     'This will clear the current session but won\'t delete the actual image files.',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                # Auto-save current annotation before clearing if enabled and there are unsaved changes
                if self.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
                    self.auto_save_current_annotation()

                # Clear all thumbnails from the grid
                for thumbnail in self.thumbnails:
                    thumbnail.deleteLater()
                self.thumbnails.clear()

                # Clear the grid layout
                for i in reversed(range(self.grid_layout.count())):
                    child = self.grid_layout.itemAt(i).widget()
                    if child:
                        child.setParent(None)

                # Reset all application state
                self.image_paths = []
                self.current_index = 0
                self.detections = []
                self.selected_boxes = set()
                self.box_labels = {}
                self.annotated_images = set()
                self.has_unsaved_changes = False

                # Clear the image display
                self.image_label.clear()
                self.image_label.setText("Load images to start")

                # Clear the label combo box (keep only model class names)
                self.label_combo.clear()
                if hasattr(self, 'class_names') and self.class_names:
                    for class_name in self.class_names.values():
                        self.label_combo.addItem(class_name)

                # Update status labels
                self.status_label.setText("Load images to start")
                self.selection_count_label.setText("Selected: 0")

                # Switch to grid view tab
                self.tab_widget.setCurrentIndex(0)

                QMessageBox.information(self, "Complete", "All images have been removed from the session.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing images: {e}")


    def remove_loaded_images(self):
        """Remove all loaded images from the thumbnail grid and reset the application state"""
        if not self.image_paths:
            QMessageBox.information(self, "Info", "No images are currently loaded.")
            return

        # # Ask for confirmation
        # reply = QMessageBox.question(self, 'Remove All Images',
        #                              'Are you sure you want to remove all loaded images?\n'
        #                              'This will clear the current session but won\'t delete the actual image files.',
        #                              QMessageBox.Yes | QMessageBox.No,
        #                              QMessageBox.No)
        #
        # if reply == QMessageBox.Yes:
        try:
            # Auto-save current annotation before clearing if enabled and there are unsaved changes
            if self.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
                self.auto_save_current_annotation()

            # Clear all thumbnails from the grid
            for thumbnail in self.thumbnails:
                thumbnail.deleteLater()
            self.thumbnails.clear()

            # Clear the grid layout
            for i in reversed(range(self.grid_layout.count())):
                child = self.grid_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)

            # Reset all application state
            self.image_paths = []
            self.current_index = 0
            self.detections = []
            self.selected_boxes = set()
            self.box_labels = {}
            self.annotated_images = set()
            self.has_unsaved_changes = False

            # Clear the image display
            self.image_label.clear()
            self.image_label.setText("Load images to start")

            # Clear the label combo box (keep only model class names)
            self.label_combo.clear()
            if hasattr(self, 'class_names') and self.class_names:
                for class_name in self.class_names.values():
                    self.label_combo.addItem(class_name)

            # Update status labels
            self.status_label.setText("Load images to start")
            self.selection_count_label.setText("Selected: 0")

            # Switch to grid view tab
            self.tab_widget.setCurrentIndex(0)

            QMessageBox.information(self, "Complete", "All images have been removed from the session.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing images: {e}")


    # Additional method to add the button to the UI
    def add_remove_images_button(self):
        """Add the remove all images button to the grid controls"""
        # This should be added to the grid_controls layout in setup_ui method
        # Find the grid_controls layout and add:
        self.remove_all_btn = QPushButton("Remove All Images")
        self.remove_all_btn.clicked.connect(self.remove_all_images)
        # grid_controls.addWidget(self.remove_all_btn)  # Add this line in setup_ui

    def export_annotations(self):
        """Export all annotated images, labels, and JSON annotations to an 'annotated_images' folder"""
        annotated_count = len([i for i in range(len(self.image_paths)) if self.has_saved_annotation(i)])

        if annotated_count == 0:
            QMessageBox.warning(self, "Warning", "No annotated images to export.")
            return

        # Ask user where to create the annotated_images folder
        base_export_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Create 'annotated_images' Folder")
        if not base_export_dir:
            return

        # Create the main export directory
        export_dir = os.path.join(base_export_dir, "annotated_images")

        try:
            # Create subdirectories
            images_dir = os.path.join(export_dir, "images")
            labels_dir = os.path.join(export_dir, "labels")
            annotations_dir = os.path.join(export_dir, "annotations")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)

            exported_count = 0
            failed_count = 0

            for i in range(len(self.image_paths)):
                if self.has_saved_annotation(i):
                    try:
                        img_path = self.image_paths[i]
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        img_ext = os.path.splitext(os.path.basename(img_path))[1]

                        # Move the original image
                        dest_img_path = os.path.join(images_dir, f"{img_name}{img_ext}")


                        # Load annotation to get the detection data
                        old_index = self.current_index
                        self.current_index = i

                        # if self.load_previous_annotation():
                            # Save YOLO format label file

                            #
                            # # Get image dimensions
                            # img = cv2.imread(img_path)
                            # h, w = img.shape[:2]
                            #
                            # with open(label_file, "w") as f:
                            #     for box_idx in self.selected_boxes:
                            #         det = self.detections[box_idx]
                            #         x1, y1, x2, y2 = det["box"]
                            #
                            #         # Determine class index
                            #         if box_idx in self.box_labels:
                            #             custom_label = self.box_labels[box_idx]
                            #             cls = None
                            #             for idx, name in self.class_names.items():
                            #                 if name == custom_label:
                            #                     cls = idx
                            #                     break
                            #             if cls is None:  # If custom label not found in class names
                            #                 cls = det["cls"]
                            #         else:
                            #             cls = det["cls"]
                            #
                            #         # Convert to YOLO format (normalized)
                            #         cx = (x1 + x2) / 2 / w
                            #         cy = (y1 + y2) / 2 / h
                            #         bw = (x2 - x1) / w
                            #         bh = (y2 - y1) / h
                            #         f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                            # Move the JSON annotation file
                        annotation_file = self.get_annotation_filename(i)
                        # if os.path.exists(labels_dir):
                        label_file = os.path.join(labels_dir, f"{img_name}.txt")
                        print(label_file)
                        if os.path.exists(annotation_file):
                            dest_annotation_path = os.path.join(annotations_dir, f"{img_name}_annotation.json")
                            shutil.move(annotation_file, dest_annotation_path)

                            dest_annotation_path = os.path.join(labels_dir, f"{img_name}.txt")
                            path = os.path.join(self.label_dir, f"{img_name}.txt")
                            shutil.move(path, dest_annotation_path)

                        exported_count += 1

                        self.current_index = old_index
                        shutil.move(img_path, dest_img_path)
                    except Exception as e:
                        print(f"Error exporting image {i}: {e}")
                        failed_count += 1
                        continue

            # Create a summary file
            summary_file = os.path.join(export_dir, "export_summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"Export Summary\n")
                f.write(f"==============\n")
                f.write(f"Export Date: {np.datetime64('now')}\n")
                f.write(f"Total Images Moved: {exported_count}\n")
                f.write(f"Failed Exports: {failed_count}\n")
                f.write(f"Total Annotated Images Found: {annotated_count}\n\n")
                f.write(f"Directory Structure:\n")
                f.write(f"- images/: Contains the original image files\n")
                f.write(f"- labels/: Contains YOLO format label files (.txt)\n")
                f.write(f"- annotations/: Contains detailed annotation files (.json)\n")

            # Also move existing YOLO label files if they exist
            for i in range(len(self.image_paths)):
                if self.has_saved_annotation(i):
                    img_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                    existing_label_file = os.path.join(self.label_dir, f"{img_name}.txt")
                    if os.path.exists(existing_label_file):
                        dest_label_path = os.path.join(labels_dir, f"{img_name}.txt")
                        if not os.path.exists(dest_label_path):  # Only move if not already created
                            shutil.move(existing_label_file, dest_label_path)

            # Show success message
            message = f"Successfully moved {exported_count} annotated images to:\n{export_dir}"
            if failed_count > 0:
                message += f"\n\nWarning: {failed_count} exports failed. Check console for details."

            QMessageBox.information(self, "Export Complete", message)

            # Ask if user wants to open the export folder
            reply = QMessageBox.question(self, 'Open Export Folder',
                                         'Do you want to open the export folder?',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                try:
                    import platform
                    if platform.system() == "Windows":
                        os.startfile(export_dir)
                    elif platform.system() == "Darwin":  # macOS
                        os.system(f"open '{export_dir}'")
                    else:  # Linux and others
                        os.system(f"xdg-open '{export_dir}'")
                except Exception as e:
                    print(f"Could not open folder: {e}")
            self.remove_loaded_images()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error during export: {e}")


    def add_images(self):
        """Add more images to the existing thumbnail grid"""
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please configure settings first.")
            self.show_settings_dialog()
            return

        # Auto-save current annotation before adding new images if enabled and there are unsaved changes
        if self.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
            self.auto_save_current_annotation()

        files, _ = QFileDialog.getOpenFileNames(self, "Select Images to Add", "", "Images (*.jpg *.png *.jpeg)")
        if files:
            # Filter out files that are already loaded to avoid duplicates
            new_files = []
            existing_files = []

            for file in files:
                if file not in self.image_paths:
                    new_files.append(file)
                else:
                    existing_files.append(os.path.basename(file))

            if not new_files:
                if existing_files:
                    QMessageBox.information(self, "Info",
                                            f"All selected images are already loaded:\n" +
                                            "\n".join(existing_files[:5]) +
                                            (f"\n... and {len(existing_files) - 5} more" if len(
                                                existing_files) > 5 else ""))
                return

            # Add new images to the existing list
            old_count = len(self.image_paths)
            self.image_paths.extend(new_files)

            # Update class names combo box if not already populated
            if not self.class_names:
                self.class_names = self.model.names
                self.label_combo.clear()
                for class_name in self.class_names.values():
                    self.label_combo.addItem(class_name)

            # Create thumbnails for the new images only
            self.add_new_thumbnails(new_files, old_count)

            # Update status
            new_count = len(new_files)
            total_count = len(self.image_paths)

            status_message = f"Added {new_count} new images. Total: {total_count} images"
            if existing_files:
                status_message += f" ({len(existing_files)} duplicates skipped)"

            self.status_label.setText(status_message)

            # Show info message
            info_message = f"Successfully added {new_count} new images."
            if existing_files:
                info_message += f"\n{len(existing_files)} duplicate(s) were skipped."

            QMessageBox.information(self, "Images Added", info_message)


    def add_new_thumbnails(self, new_files, start_index):
        """Create thumbnails for newly added images and add them to the grid"""
        cols = 5  # Number of columns in grid (should match the value in create_grid_thumbnails)

        for i, img_path in enumerate(new_files):
            thumbnail_index = start_index + i
            thumbnail = ImageThumbnail(img_path, thumbnail_index)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)

            # Calculate grid position
            row = thumbnail_index // cols
            col = thumbnail_index % cols
            self.grid_layout.addWidget(thumbnail, row, col)
            self.thumbnails.append(thumbnail)

            # Check if this image already has a saved annotation
            annotation_exists = self.has_saved_annotation(thumbnail_index)
            thumbnail.set_annotated(annotation_exists)


    def update_all_thumbnail_indices(self):
        """Update all thumbnail indices after images are added or removed"""
        for i, thumbnail in enumerate(self.thumbnails):
            thumbnail.index = i
            # Update the annotation status
            annotation_exists = self.has_saved_annotation(i)
            thumbnail.set_annotated(annotation_exists)

    def remove_selected_images(self):
        """Remove selected images from the thumbnail grid"""
        if not self.image_paths:
            QMessageBox.information(self, "Info", "No images are currently loaded.")
            return

        # Get selected thumbnails
        selected_indices = []
        for i, thumbnail in enumerate(self.thumbnails):
            if hasattr(thumbnail, 'is_selected') and thumbnail.is_selected:
                selected_indices.append(i)

        if not selected_indices:
            QMessageBox.information(self, "Info", "No images are selected for removal.")
            return

        # Ask for confirmation
        reply = QMessageBox.question(self, 'Remove Selected Images',
                                     f'Are you sure you want to remove {len(selected_indices)} selected image(s)?\n'
                                     'This will remove them from the current session but won\'t delete the actual image files.',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                # Auto-save current annotation before removing if enabled and there are unsaved changes
                if (self.auto_save_enabled and self.current_index in selected_indices and
                        self.detections and self.has_unsaved_changes):
                    self.auto_save_current_annotation()

                # Sort indices in descending order to remove from end to beginning
                # This prevents index shifting issues
                selected_indices.sort(reverse=True)

                # Remove images and thumbnails
                removed_count = 0
                for index in selected_indices:
                    if 0 <= index < len(self.image_paths):
                        # Remove from image paths
                        self.image_paths.pop(index)

                        # Remove thumbnail widget
                        thumbnail = self.thumbnails.pop(index)
                        thumbnail.deleteLater()

                        # Update annotated images set (shift indices down for removed images)
                        new_annotated = set()
                        for ann_idx in self.annotated_images:
                            if ann_idx < index:
                                new_annotated.add(ann_idx)
                            elif ann_idx > index:
                                new_annotated.add(ann_idx - 1)
                            # Skip ann_idx == index (the removed image)
                        self.annotated_images = new_annotated

                        removed_count += 1

                # Clear the grid layout
                for i in reversed(range(self.grid_layout.count())):
                    child = self.grid_layout.itemAt(i).widget()
                    if child:
                        child.setParent(None)

                # Recreate the grid with remaining thumbnails
                if self.image_paths:
                    # Update thumbnail indices and recreate grid
                    cols = 5
                    for i, thumbnail in enumerate(self.thumbnails):
                        thumbnail.index = i
                        row = i // cols
                        col = i % cols
                        self.grid_layout.addWidget(thumbnail, row, col)

                        # Update annotation status
                        annotation_exists = self.has_saved_annotation(i)
                        thumbnail.set_annotated(annotation_exists)
                        # Clear selection state
                        thumbnail.set_selected(False)

                    # Adjust current index if necessary
                    if self.current_index >= len(self.image_paths):
                        self.current_index = len(self.image_paths) - 1
                    elif self.current_index < 0:
                        self.current_index = 0

                    # Update current thumbnail state
                    if self.thumbnails and 0 <= self.current_index < len(self.thumbnails):
                        self.thumbnails[self.current_index].set_current(True)

                    # Refresh current image display
                    self.show_current_image()
                else:
                    # No images left, reset everything
                    self.current_index = 0
                    self.detections = []
                    self.selected_boxes = set()
                    self.box_labels = {}
                    self.annotated_images = set()
                    self.has_unsaved_changes = False

                    # Clear the image display
                    self.image_label.clear()
                    self.image_label.setText("Load images to start")

                    # Update status labels
                    self.status_label.setText("Load images to start")
                    self.selection_count_label.setText("Selected: 0")

                QMessageBox.information(self, "Complete",
                                        f"Successfully removed {removed_count} image(s). "
                                        f"Remaining images: {len(self.image_paths)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing selected images: {e}")

    def select_all_thumbnails(self):
        """Select all thumbnails in the grid"""
        if not self.thumbnails:
            return

        for thumbnail in self.thumbnails:
            thumbnail.set_selected(True)

        selected_count = len(self.thumbnails)
        QMessageBox.information(self, "Selection", f"Selected all {selected_count} images.")

    def deselect_all_thumbnails(self):
        """Deselect all thumbnails in the grid"""
        if not self.thumbnails:
            return

        for thumbnail in self.thumbnails:
            thumbnail.set_selected(False)

        QMessageBox.information(self, "Selection", "Deselected all images.")

    def get_selected_thumbnail_count(self):
        """Get the number of selected thumbnails"""
        if not self.thumbnails:
            return 0

        count = 0
        for thumbnail in self.thumbnails:
            if hasattr(thumbnail, 'is_selected') and thumbnail.is_selected:
                count += 1
        return count

    def update_thumbnail_selection_ui(self):
        """Update UI elements related to thumbnail selection"""
        selected_count = self.get_selected_thumbnail_count()
        if hasattr(self, 'thumbnail_selection_label'):
            self.thumbnail_selection_label.setText(f"Selected: {selected_count} / {len(self.thumbnails)}")

    # Add these buttons to your setup_ui method in the grid_controls section:

