import cv2
import math
import os
import json
import time
import numpy as np
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPen, QPainter, QColor, QFont, QIntValidator
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QGridLayout, QFrame, QMessageBox,
    QCheckBox, QSlider, QSpinBox, QLineEdit, QTextEdit
)
from ultralytics import YOLO


class CameraAnnotation(QWidget):
    def __init__(self, parent_annotator=None):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.available_cameras = []
        self.camera_feeds = {}  # {camera_id: (cap, QLabel, QFrame, QLabel)}
        self.timers = {}
        self.active_cameras = []  # Track order of added cameras

        # Detection settings
        self.model = None
        self.detection_enabled = False
        self.confidence_threshold = 0.5
        self.class_names = {}
        self.capture_count = 0

        # New features for class-specific annotation
        self.continuous_capture = False
        self.target_class_continuous = ""
        self.wrong_detection_mode = False
        self.target_class_wrong = ""
        self.save_directory = "annotations"

        # Initialize components
        try:
            self.create_save_directory()
            self.detect_cameras_safe()
            self.load_model_settings()
            self.setup_ui()
        except Exception as e:
            print(f"Initialization error: {e}")
            self.show_error_message("Initialization Error", f"Failed to initialize: {e}")

    def show_error_message(self, title, message):
        """Safely show error message"""
        try:
            QMessageBox.warning(self, title, message)
        except:
            print(f"{title}: {message}")

    def create_save_directory(self):
        """Create directory structure for saving annotations"""
        try:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)

            # Create subdirectories
            subdirs = ["continuous_capture", "wrong_detections", "images", "labels", "yolo_labels"]
            for subdir in subdirs:
                path = os.path.join(self.save_directory, subdir)
                if not os.path.exists(path):
                    os.makedirs(path)
        except Exception as e:
            print(f"Error creating save directory: {e}")

    def detect_cameras_safe(self):
        """Safely detect cameras without crashing"""
        self.available_cameras = []
        try:
            for i in range(5):  # Reduced range to prevent crashes
                try:
                    cap = cv2.VideoCapture(i)
                    if cap is not None and cap.isOpened():
                        # Test if we can read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.available_cameras.append(i)
                    cap.release()
                except Exception as e:
                    print(f"Error testing camera {i}: {e}")
                    continue
        except Exception as e:
            print(f"Camera detection error: {e}")

    def load_model_settings(self):
        """Load YOLO model from settings manager"""
        try:
            if (self.parent_annotator and
                    hasattr(self.parent_annotator, 'settings_manager') and
                    self.parent_annotator.settings_manager is not None):

                settings_manager = self.parent_annotator.settings_manager

                if settings_manager.settings_exist():
                    try:
                        settings = settings_manager.load_settings()
                        model_path = settings.get('model_path')

                        if model_path and os.path.exists(model_path):
                            self.model = YOLO(model_path)
                            if hasattr(self.model, 'names'):
                                self.class_names = self.model.names
                                print(f"Loaded YOLO model: {model_path}")
                                print(f"Available classes: {len(self.class_names)}")
                        else:
                            print("Model path not found in settings")
                    except Exception as e:
                        print(f"Error loading model: {e}")
        except Exception as e:
            print(f"load_model_settings error: {e}")

    def setup_ui(self):
        """Setup user interface with error handling"""
        try:
            main_layout = QHBoxLayout()

            # Left panel
            left_panel = QVBoxLayout()
            left_widget = QWidget()
            left_widget.setMaximumWidth(400)
            left_widget.setLayout(left_panel)

            # Camera selection group
            self.setup_camera_selection(left_panel)

            # Detection settings group (only if model is loaded)
            if self.model is not None:
                self.setup_detection_settings(left_panel)
                self.setup_continuous_capture(left_panel)
                self.setup_wrong_detection(left_panel)
                self.setup_log_display(left_panel)
            else:
                self.setup_no_model_message(left_panel)

            # Active cameras group
            self.setup_active_cameras(left_panel)

            # Status
            self.camera_status = QLabel("Select a camera and click Add to start")
            left_panel.addWidget(self.camera_status)
            left_panel.addStretch()

            # Right panel for camera feeds
            self.setup_camera_feeds()

            main_layout.addWidget(left_widget)
            main_layout.addWidget(self.scroll_area)
            self.setLayout(main_layout)

        except Exception as e:
            print(f"UI setup error: {e}")

    def setup_camera_selection(self, parent_layout):
        """Setup camera selection controls"""
        camera_selection = QGroupBox("Add Cameras")
        selection_layout = QVBoxLayout()

        # Combo box and add button
        combo_layout = QHBoxLayout()
        self.camera_combo = QComboBox()


        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_selected_camera)

        combo_layout.addWidget(self.camera_combo)
        combo_layout.addWidget(self.add_button)
        selection_layout.addLayout(combo_layout)

        # Refresh button
        self.refresh_button = QPushButton("Refresh Cameras")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        selection_layout.addWidget(self.refresh_button)

        camera_selection.setLayout(selection_layout)
        parent_layout.addWidget(camera_selection)
        self.populate_camera_combo()

    def setup_detection_settings(self, parent_layout):
        """Setup detection settings controls"""
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()

        # Enable/disable detection
        self.detection_checkbox = QCheckBox("Enable Detection")
        self.detection_checkbox.stateChanged.connect(self.toggle_detection)
        detection_layout.addWidget(self.detection_checkbox)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_confidence)

        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setMinimum(10)
        self.conf_spinbox.setMaximum(95)
        self.conf_spinbox.setValue(50)
        self.conf_spinbox.valueChanged.connect(self.conf_slider.setValue)
        self.conf_slider.valueChanged.connect(self.conf_spinbox.setValue)

        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spinbox)
        conf_layout.addWidget(QLabel("%"))
        detection_layout.addLayout(conf_layout)

        # Model info
        model_info = QLabel(f"Classes: {len(self.class_names)}")
        model_info.setStyleSheet("color: green; font-size: 10px;")
        detection_layout.addWidget(model_info)

        detection_group.setLayout(detection_layout)
        parent_layout.addWidget(detection_group)

    def setup_continuous_capture(self, parent_layout):
        """Setup continuous capture controls with class selection"""
        continuous_group = QGroupBox("Function 1: Continuous Class Capture")
        continuous_layout = QVBoxLayout()

        # Class selection - ComboBox + LineEdit for custom input
        class_selection_layout = QVBoxLayout()

        # ComboBox with model classes
        class_combo_layout = QHBoxLayout()
        class_combo_layout.addWidget(QLabel("Select Class:"))
        self.continuous_class_combo = QComboBox()
        self.continuous_class_combo.setEditable(True)
        self.populate_class_combo(self.continuous_class_combo)
        self.continuous_class_combo.currentTextChanged.connect(self.on_continuous_class_changed)
        class_combo_layout.addWidget(self.continuous_class_combo)
        class_selection_layout.addLayout(class_combo_layout)

        # Custom class input
        custom_class_layout = QHBoxLayout()
        custom_class_layout.addWidget(QLabel("Or enter custom:"))
        self.continuous_class_input = QLineEdit()
        self.continuous_class_input.setPlaceholderText("Enter class name (e.g., person, car)")
        self.continuous_class_input.textChanged.connect(self.on_continuous_custom_changed)
        custom_class_layout.addWidget(self.continuous_class_input)
        class_selection_layout.addLayout(custom_class_layout)

        continuous_layout.addLayout(class_selection_layout)

        # Capture Limit input
        capture_limit_layout = QHBoxLayout()
        capture_limit_layout.addWidget(QLabel("Capture Limit: "))
        self.capture_limit_input = QLineEdit()
        self.capture_limit_input.setPlaceholderText("0 - 10000")
        # self.capture_limit_input.textChanged.connect(self.on_continuous_custom_changed)
        capture_limit_layout.addWidget(self.capture_limit_input)
        class_selection_layout.addLayout(capture_limit_layout)

        # Allow only integers from 0 to 10000
        int_validator = QIntValidator(0, 10000)
        self.capture_limit_input.setValidator(int_validator)
        continuous_layout.addLayout(class_selection_layout)

        # Buttons
        self.start_continuous_btn = QPushButton("Start Continuous Capture")
        self.start_continuous_btn.clicked.connect(self.start_continuous_capture)
        self.stop_continuous_btn = QPushButton("Stop Continuous Capture")
        self.stop_continuous_btn.clicked.connect(self.stop_continuous_capture)
        self.stop_continuous_btn.setEnabled(False)

        continuous_buttons_layout = QHBoxLayout()
        continuous_buttons_layout.addWidget(self.start_continuous_btn)
        continuous_buttons_layout.addWidget(self.stop_continuous_btn)
        continuous_layout.addLayout(continuous_buttons_layout)

        # Status
        self.continuous_status = QLabel("Status: Ready")
        self.continuous_status.setStyleSheet("color: blue; font-size: 10px;")
        continuous_layout.addWidget(self.continuous_status)

        continuous_group.setLayout(continuous_layout)
        parent_layout.addWidget(continuous_group)

    def setup_wrong_detection(self, parent_layout):
        """Setup wrong detection controls with class selection"""
        wrong_detection_group = QGroupBox("Function 2: Wrong Detection Capture")
        wrong_layout = QVBoxLayout()

        # Class selection - ComboBox + LineEdit for custom input
        class_selection_layout = QVBoxLayout()

        # ComboBox with model classes
        class_combo_layout = QHBoxLayout()
        class_combo_layout.addWidget(QLabel("Expected Class:"))
        self.wrong_class_combo = QComboBox()
        self.wrong_class_combo.setEditable(True)
        self.populate_class_combo(self.wrong_class_combo)
        self.wrong_class_combo.currentTextChanged.connect(self.on_wrong_class_changed)
        class_combo_layout.addWidget(self.wrong_class_combo)
        class_selection_layout.addLayout(class_combo_layout)

        # Custom class input
        custom_class_layout = QHBoxLayout()
        custom_class_layout.addWidget(QLabel("Or enter custom:"))
        self.wrong_class_input = QLineEdit()
        self.wrong_class_input.setPlaceholderText("Enter expected class name")
        self.wrong_class_input.textChanged.connect(self.on_wrong_custom_changed)
        custom_class_layout.addWidget(self.wrong_class_input)
        class_selection_layout.addLayout(custom_class_layout)

        wrong_layout.addLayout(class_selection_layout)

        # Buttons
        self.start_wrong_detection_btn = QPushButton("Start Wrong Detection Mode")
        self.start_wrong_detection_btn.clicked.connect(self.start_wrong_detection_mode)
        self.stop_wrong_detection_btn = QPushButton("Stop Wrong Detection Mode")
        self.stop_wrong_detection_btn.clicked.connect(self.stop_wrong_detection_mode)
        self.stop_wrong_detection_btn.setEnabled(False)

        wrong_buttons_layout = QHBoxLayout()
        wrong_buttons_layout.addWidget(self.start_wrong_detection_btn)
        wrong_buttons_layout.addWidget(self.stop_wrong_detection_btn)
        wrong_layout.addLayout(wrong_buttons_layout)

        # Status
        self.wrong_detection_status = QLabel("Status: Ready")
        self.wrong_detection_status.setStyleSheet("color: orange; font-size: 10px;")
        wrong_layout.addWidget(self.wrong_detection_status)

        wrong_detection_group.setLayout(wrong_layout)
        parent_layout.addWidget(wrong_detection_group)

    def populate_class_combo(self, combo_box):
        """Populate combo box with model classes"""
        try:
            combo_box.clear()
            combo_box.addItem("")  # Empty option for custom input

            if self.class_names:
                # Sort class names for better usability
                print("CHECK THE VALUE:!!!!!!!!!",self.class_names.values())
                sorted_classes = sorted(self.class_names.values())
                for class_name in sorted_classes:
                    combo_box.addItem(class_name)
        except Exception as e:
            print(f"Error populating class combo: {e}")

    def on_continuous_class_changed(self, text):
        """Handle continuous class combo selection"""
        if text and text != self.continuous_class_input.text():
            self.continuous_class_input.setText(text)

    def on_continuous_custom_changed(self, text):
        """Handle continuous custom class input"""
        if text and text != self.continuous_class_combo.currentText():
            self.continuous_class_combo.setCurrentText(text)

    def on_wrong_class_changed(self, text):
        """Handle wrong detection class combo selection"""
        if text and text != self.wrong_class_input.text():
            self.wrong_class_input.setText(text)

    def on_wrong_custom_changed(self, text):
        """Handle wrong detection custom class input"""
        if text and text != self.wrong_class_combo.currentText():
            self.wrong_class_combo.setCurrentText(text)

    def setup_log_display(self, parent_layout):
        """Setup log display"""
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)

    def setup_no_model_message(self, parent_layout):
        """Setup no model message"""
        no_model_label = QLabel("No YOLO model loaded.\nConfigure model in main settings.")
        no_model_label.setStyleSheet("color: orange; font-style: italic;")
        no_model_label.setAlignment(Qt.AlignCenter)
        parent_layout.addWidget(no_model_label)

    def setup_active_cameras(self, parent_layout):
        """Setup active cameras display"""
        active_cameras = QGroupBox("Active Cameras")
        active_layout = QVBoxLayout()

        self.active_cameras_list = QLabel("No cameras active")
        active_layout.addWidget(self.active_cameras_list)

        self.remove_all_button = QPushButton("Remove All Cameras")
        self.remove_all_button.clicked.connect(self.remove_all_cameras)
        self.remove_all_button.setEnabled(False)
        active_layout.addWidget(self.remove_all_button)

        active_cameras.setLayout(active_layout)
        parent_layout.addWidget(active_cameras)

    def setup_camera_feeds(self):
        """Setup camera feeds display area"""
        self.feed_layout = QGridLayout()
        self.feed_layout.setSpacing(10)
        self.feed_container = QWidget()
        self.feed_container.setLayout(self.feed_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.feed_container)

    def log_message(self, message):
        """Add message to log display"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            if hasattr(self, 'log_display'):
                self.log_display.append(log_entry)
            print(log_entry)
        except Exception as e:
            print(f"Log error: {e}")

    def get_class_id_for_name(self, class_name):
        """Get class ID for a given class name"""
        try:
            class_name_lower = class_name.lower()

            # First, try exact match in model class names
            for cls_id, cls_name in self.class_names.items():
                if cls_name.lower() == class_name_lower:
                    return cls_id

            # If not found in model classes, return the next available ID
            max_id = max(self.class_names.keys()) if self.class_names else -1
            return max_id + 1
        except Exception as e:
            print(f"Error getting class ID: {e}")
            return 0

    def save_yolo_format(self, detections, image_shape, filename_base, target_class):
        """Save annotations in YOLO format"""
        try:
            h, w = image_shape[:2]
            label_file = os.path.join(self.save_directory, "yolo_labels", f"{filename_base}.txt")

            with open(label_file, "w") as f:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]

                    # Get class ID
                    if det["class_name"].lower() == target_class.lower():
                        # Use the model's class ID if it matches
                        cls = det["cls"]
                    else:
                        # Use custom class ID for the target class
                        cls = self.get_class_id_for_name(target_class)

                    # Convert to YOLO format (normalized)
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            return True
        except Exception as e:
            print(f"Error saving YOLO format: {e}")
            return False

    def save_current_annotation(self, frame, detections, camera_id, annotation_type, target_class):
        """Save current annotation state to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename_base = f"cam{camera_id}_{timestamp}"

            # Save image
            image_path = os.path.join(self.save_directory, "images", f"{filename_base}.jpg")
            success = cv2.imwrite(image_path, frame)
            if not success:
                print(f"Failed to save image: {image_path}")
                return False

            # Prepare selected boxes (all detections that match target class)
            selected_boxes = []
            box_labels = {}

            for i, det in enumerate(detections):
                if det["class_name"].lower() == target_class.lower():
                    selected_boxes.append(i)
                    box_labels[i] = target_class

            annotation_data = {
                "image_path": image_path,
                "detections": detections,
                "selected_boxes": selected_boxes,
                "box_labels": box_labels,
                "original_shape": frame.shape,
                "timestamp": str(np.datetime64('now')),
                "camera_id": camera_id,
                "annotation_type": annotation_type,
                "target_class": target_class,
                "image_dimensions": {'width': frame.shape[1], 'height': frame.shape[0]}
            }

            # Save main annotation JSON
            annotation_file = os.path.join(self.save_directory, "labels", f"{filename_base}.json")
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)

            # Also save in specific subdirectory
            subdir = "continuous_capture" if annotation_type == "continuous" else "wrong_detections"
            subdir_json_path = os.path.join(self.save_directory, subdir, f"{filename_base}_annotation.json")
            with open(subdir_json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)

            return filename_base
        except Exception as e:
            print(f"Error saving annotation: {e}")
            return None

    def save_annotations(self, frame, detections, camera_id, annotation_type, target_class):
        """Save both YOLO format and annotation JSON"""
        try:
            if not detections:
                return False

            # Save annotation JSON and get filename
            filename_base = self.save_current_annotation(frame, detections, camera_id, annotation_type, target_class)
            if not filename_base:
                return False

            # Save YOLO format
            success = self.save_yolo_format(detections, frame.shape, filename_base, target_class)

            return success
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False

    def start_continuous_capture(self):
        """Function 1: Start continuous capture for specific class"""
        try:
            target_class = self.continuous_class_input.text().strip()
            if not target_class:
                self.show_error_message("Warning", "Please enter a target class name")
                return

            if not self.model:
                self.show_error_message("Warning", "No YOLO model loaded")
                return

            if not self.active_cameras:
                self.show_error_message("Warning", "No active cameras")
                return

            self.continuous_capture = True
            self.target_class_continuous = target_class
            self.detection_enabled = True

            if hasattr(self, 'detection_checkbox'):
                self.detection_checkbox.setChecked(True)

            self.start_continuous_btn.setEnabled(False)
            self.stop_continuous_btn.setEnabled(True)
            self.continuous_status.setText(f"Status: Capturing '{target_class}' objects")
            self.continuous_status.setStyleSheet("color: green; font-size: 10px;")

            self.log_message(f"Started continuous capture for class: {target_class}")
        except Exception as e:
            print(f"Error starting continuous capture: {e}")

    def stop_continuous_capture(self):
        """Stop continuous capture"""
        try:
            self.continuous_capture = False
            self.target_class_continuous = ""

            self.start_continuous_btn.setEnabled(True)
            self.stop_continuous_btn.setEnabled(False)
            self.continuous_status.setText("Status: Ready")
            self.continuous_status.setStyleSheet("color: blue; font-size: 10px;")

            self.log_message("Stopped continuous capture")
        except Exception as e:
            print(f"Error stopping continuous capture: {e}")

    def start_wrong_detection_mode(self):
        """Function 2: Start wrong detection capture mode"""
        try:
            expected_class = self.wrong_class_input.text().strip()
            if not expected_class:
                self.show_error_message("Warning", "Please enter expected class name")
                return

            if not self.model:
                self.show_error_message("Warning", "No YOLO model loaded")
                return

            if not self.active_cameras:
                self.show_error_message("Warning", "No active cameras")
                return

            self.wrong_detection_mode = True
            self.target_class_wrong = expected_class
            self.detection_enabled = True

            if hasattr(self, 'detection_checkbox'):
                self.detection_checkbox.setChecked(True)

            self.start_wrong_detection_btn.setEnabled(False)
            self.stop_wrong_detection_btn.setEnabled(True)
            self.wrong_detection_status.setText(f"Status: Monitoring for wrong detections of '{expected_class}'")
            self.wrong_detection_status.setStyleSheet("color: red; font-size: 10px;")

            self.log_message(f"Started wrong detection mode for expected class: {expected_class}")
        except Exception as e:
            print(f"Error starting wrong detection mode: {e}")

    def stop_wrong_detection_mode(self):
        """Stop wrong detection mode"""
        try:
            self.wrong_detection_mode = False
            self.target_class_wrong = ""

            self.start_wrong_detection_btn.setEnabled(True)
            self.stop_wrong_detection_btn.setEnabled(False)
            self.wrong_detection_status.setText("Status: Ready")
            self.wrong_detection_status.setStyleSheet("color: orange; font-size: 10px;")

            self.log_message("Stopped wrong detection mode")
        except Exception as e:
            print(f"Error stopping wrong detection mode: {e}")

    def toggle_detection(self, state):
        """Enable or disable detection"""
        try:
            self.detection_enabled = state == Qt.Checked
            status = "enabled" if self.detection_enabled else "disabled"
            if hasattr(self, 'camera_status'):
                self.camera_status.setText(f"Detection {status}")
        except Exception as e:
            print(f"toggle_detection error: {e}")

    def update_confidence(self, value):
        """Update confidence threshold"""
        try:
            self.confidence_threshold = value / 100.0
        except Exception as e:
            print(f"update_confidence error: {e}")

    def detect_objects(self, frame, camera_id):
        """Run YOLO detection on frame with enhanced annotation features"""
        if not self.model or not self.detection_enabled:
            return frame, []

        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = []
            annotated_frame = frame.copy()


            if results and len(results) > 0:
                result = results[0]
                # if self.class_names:
                #     # Sort class names for better usability
                #     print("CHECK THE VALUE:!!!!!!!!!", self.class_names.values())
                #     sorted_classes = sorted(self.class_names.values())
                #     for idx, class_names in sorted_classes:
                #         if idx == int(cls_id):
                #             class_id = int(cls_id)
                #         else:
                #             class_id = 0
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        area = (int(x2) - int(x1)) * (int(y2) - int(y1))
                        # if target_detections in values_list:
                        index = list(self.class_names.values()).index(self.target_class_continuous.lower())
                        class_name = self.class_names.get(index, f"Class {index}")


                        detections.append({
                            'cls': index,  # For compatibility
                            'conf': float(conf),
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            "area": area,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                             # Alternative format for compatibility
                            'class_name': class_name
                        })

                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label with confidence
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                        # Draw label background
                        cv2.rectangle(annotated_frame,
                                      (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1),
                                      (0, 255, 0), -1)

                        # Draw label text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            limit_text = self.capture_limit_input.text()
            capture_limit = int(limit_text) if limit_text.isdigit() else 0

            # Function 1: Continuous capture for specific class
            if self.continuous_capture and self.target_class_continuous and detections and (capture_limit == 0 or self.capture_count < capture_limit):
                try:

                    target_detections = [det for det in detections
                                         if det['class_name'].lower() == self.target_class_continuous.lower()]

                    detected_classes = [det['class_name'].lower() for det in detections]
                    expected_class_lower = self.target_class_continuous.lower()

                    # Check if any detection is NOT the expected class
                    wrong_detections = [cls for cls in detected_classes if cls != expected_class_lower]
                    self.log_message(f"Target {self.target_class_continuous.lower()}")
                    if target_detections or wrong_detections:

                        if self.save_annotations(frame, detections, camera_id, "continuous",
                                                 self.target_class_continuous):
                            self.log_message(
                                f"Saved continuous capture: {len(target_detections)} '{self.target_class_continuous}' objects detected on camera {camera_id}")

                        else:
                            self.log_message(f"Failed to save continuous capture for camera {camera_id}")
                    self.capture_count += 1
                    if self.capture_count == capture_limit:
                        QMessageBox.information(self, "Info", "Capture limit has been reached!")
                except Exception as e:
                    print(f"Continuous capture error: {e}")

            # Function 2: Wrong detection capture
            if self.wrong_detection_mode and self.target_class_wrong and detections:
                try:
                    detected_classes = [det['class_name'].lower() for det in detections]
                    expected_class_lower = self.target_class_wrong.lower()

                    # Check if any detection is NOT the expected class
                    wrong_detections = [cls for cls in detected_classes if cls != expected_class_lower]
                    if wrong_detections and detected_classes:  # Only save if there are detections
                        if self.save_annotations(frame, detections, camera_id, "wrong_detection",
                                                 self.target_class_wrong):
                            wrong_classes_str = ", ".join(set(wrong_detections))
                            self.log_message(
                                f"Saved wrong detection: Expected '{self.target_class_wrong}', got '{wrong_classes_str}' on camera {camera_id}")
                        else:
                            self.log_message(f"Failed to save wrong detection for camera {camera_id}")
                except Exception as e:
                    print(f"Wrong detection error: {e}")

            return annotated_frame, detections

        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []

    def populate_camera_combo(self):
        """Populate combo box with available cameras"""
        try:
            self.camera_combo.clear()
            if not self.available_cameras:
                self.camera_combo.addItem("No cameras detected")
                self.add_button.setEnabled(False)
            else:
                available_to_add = [cam_id for cam_id in self.available_cameras if cam_id not in self.active_cameras]
                if not available_to_add:
                    self.camera_combo.addItem("All cameras in use")
                    self.add_button.setEnabled(False)
                else:
                    for cam_id in available_to_add:
                        self.camera_combo.addItem(f"Camera {cam_id}", cam_id)
                    self.add_button.setEnabled(True)
        except Exception as e:
            print(f"populate_camera_combo error: {e}")

    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        try:
            self.detect_cameras_safe()
            self.populate_camera_combo()
            self.camera_status.setText(f"Found {len(self.available_cameras)} cameras")
        except Exception as e:
            print(f"refresh_cameras error: {e}")

    def add_selected_camera(self):
        """Add the selected camera from combo box"""
        try:
            if self.camera_combo.count() == 0 or not self.add_button.isEnabled():
                return

            cam_id = self.camera_combo.currentData()
            if cam_id is None:
                return

            if cam_id in self.active_cameras:
                self.show_error_message("Warning", f"Camera {cam_id} is already active!")
                return

            self.start_camera(cam_id)
        except Exception as e:
            print(f"add_selected_camera error: {e}")

    def start_camera(self, cam_id):
        """Start a camera feed"""
        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap or not cap.isOpened():
                self.camera_status.setText(f"Failed to open camera {cam_id}")
                self.show_error_message("Error", f"Failed to open camera {cam_id}")
                if cap:
                    cap.release()
                return

            # Test reading a frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                self.show_error_message("Error", f"Camera {cam_id} cannot provide frames")
                cap.release()
                return

            # Create feed widget
            feed_widget = QFrame()
            feed_widget.setFrameStyle(QFrame.Box)
            feed_layout = QVBoxLayout()

            title_label = QLabel(f"Camera {cam_id}")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; padding: 5px;")

            video_label = QLabel()
            video_label.setMinimumSize(320, 240)
            video_label.setStyleSheet("border: 1px solid gray;")
            video_label.setAlignment(Qt.AlignCenter)
            video_label.setText("Loading...")

            detection_info = QLabel("Detection: OFF")
            detection_info.setStyleSheet("font-size: 10px; color: gray;")
            detection_info.setAlignment(Qt.AlignCenter)

            remove_button = QPushButton("Remove")
            remove_button.clicked.connect(lambda: self.remove_camera(cam_id))

            feed_layout.addWidget(title_label)
            feed_layout.addWidget(video_label)
            feed_layout.addWidget(detection_info)
            feed_layout.addWidget(remove_button)
            feed_widget.setLayout(feed_layout)

            # Add to grid
            self.reorganize_grid()
            num_cameras = len(self.active_cameras)
            cols = max(1, math.ceil(math.sqrt(num_cameras + 1)))
            row = num_cameras // cols
            col = num_cameras % cols
            self.feed_layout.addWidget(feed_widget, row, col)

            # Store camera data
            self.camera_feeds[cam_id] = (cap, video_label, feed_widget, detection_info)
            self.active_cameras.append(cam_id)

            # Setup timer
            timer = QTimer(self)
            timer.timeout.connect(lambda: self.update_frame(cam_id))
            timer.start(50)  # 20 FPS to reduce load
            self.timers[cam_id] = timer

            self.camera_status.setText(f"Camera {cam_id} started ({len(self.active_cameras)} total)")
            self.update_active_cameras_display()
            self.populate_camera_combo()
            self.remove_all_button.setEnabled(True)

        except Exception as e:
            print(f"start_camera error: {e}")
            if 'cap' in locals() and cap:
                cap.release()

    def remove_camera(self, cam_id):
        """Remove a specific camera"""
        try:
            if cam_id in self.camera_feeds:
                cap, video_label, feed_widget, detection_info = self.camera_feeds.pop(cam_id)
                if cap:
                    cap.release()
                if feed_widget:
                    feed_widget.deleteLater()

            if cam_id in self.timers:
                self.timers[cam_id].stop()
                del self.timers[cam_id]

            if cam_id in self.active_cameras:
                self.active_cameras.remove(cam_id)

            self.camera_status.setText(f"Camera {cam_id} removed ({len(self.active_cameras)} remaining)")
            self.reorganize_grid()
            self.update_active_cameras_display()
            self.populate_camera_combo()

            if len(self.active_cameras) == 0:
                self.remove_all_button.setEnabled(False)

        except Exception as e:
            print(f"remove_camera error: {e}")

    def remove_all_cameras(self):
        """Remove all active cameras"""
        try:
            cameras_to_remove = self.active_cameras.copy()
            for cam_id in cameras_to_remove:
                self.remove_camera(cam_id)
        except Exception as e:
            print(f"remove_all_cameras error: {e}")

    def reorganize_grid(self):
        """Reorganize cameras in optimal grid layout"""
        try:
            if not self.active_cameras:
                return

            # Clear current layout
            for i in reversed(range(self.feed_layout.count())):
                item = self.feed_layout.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.setParent(None)

            # Calculate optimal grid dimensions
            num_cameras = len(self.active_cameras)
            cols = max(1, math.ceil(math.sqrt(num_cameras)))

            # Redistribute cameras
            for i, cam_id in enumerate(self.active_cameras):
                if cam_id in self.camera_feeds:
                    _, _, feed_widget, _ = self.camera_feeds[cam_id]
                    if feed_widget:
                        row = i // cols
                        col = i % cols
                        self.feed_layout.addWidget(feed_widget, row, col)
        except Exception as e:
            print(f"reorganize_grid error: {e}")

    def update_active_cameras_display(self):
        """Update the active cameras list display"""
        try:
            if not self.active_cameras:
                self.active_cameras_list.setText("No cameras active")
            else:
                camera_list = ", ".join([f"Camera {cam_id}" for cam_id in self.active_cameras])
                self.active_cameras_list.setText(f"Active: {camera_list}")
        except Exception as e:
            print(f"update_active_cameras_display error: {e}")

    def update_frame(self, cam_id):
        """Update frame for a specific camera"""
        try:
            if cam_id not in self.camera_feeds:
                return

            cap, video_label, _, detection_info = self.camera_feeds[cam_id]

            if not cap or not cap.isOpened():
                video_label.setText(f"Camera {cam_id} disconnected")
                detection_info.setText("Disconnected")
                detection_info.setStyleSheet("font-size: 10px; color: red;")
                return

            ret, frame = cap.read()
            if not ret or frame is None:
                video_label.setText(f"Camera {cam_id} no signal")
                detection_info.setText("No Signal")
                detection_info.setStyleSheet("font-size: 10px; color: red;")
                return

            # Run detection if enabled
            if self.detection_enabled and self.model:
                try:
                    annotated_frame, detections = self.detect_objects(frame, cam_id)
                    detection_count = len(detections)
                    detection_info.setText(f"Objects: {detection_count}")
                    detection_info.setStyleSheet("font-size: 10px; color: green;")
                except Exception as e:
                    print(f"Detection error for camera {cam_id}: {e}")
                    annotated_frame = frame
                    detection_info.setText("Detection Error")
                    detection_info.setStyleSheet("font-size: 10px; color: red;")
            else:
                annotated_frame = frame
                detection_info.setText("Detection: OFF")
                detection_info.setStyleSheet("font-size: 10px; color: gray;")

            # Convert to RGB and display
            try:
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                # Scale to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                video_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"Frame display error for camera {cam_id}: {e}")
                video_label.setText(f"Display Error")

        except Exception as e:
            print(f"update_frame error for camera {cam_id}: {e}")

    def closeEvent(self, event):
        """Clean up when closing the application"""
        try:
            # Stop all annotation modes
            if hasattr(self, 'continuous_capture') and self.continuous_capture:
                self.stop_continuous_capture()
            if hasattr(self, 'wrong_detection_mode') and self.wrong_detection_mode:
                self.stop_wrong_detection_mode()

            # Remove all cameras
            self.remove_all_cameras()
            event.accept()
        except Exception as e:
            print(f"closeEvent error: {e}")
            event.accept()

