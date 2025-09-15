import os
import cv2
import json
import shutil
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QTabWidget, QDialog
)
from PyQt5.QtWidgets import QCompleter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from main.tab_view import (
    ImageThumbnail,
    CustomLabelManager,
    CustomLabelDialog,
    DetailView,
    GridView,
    PreprocessView,
    CameraAnnotation,
)
from main.settings import SettingsDialog, SettingsManager
# from main.tab_view.t_grid_view.image_thumbnail import ImageThumbnail
# from tab_view import CustomLabelManager, CustomLabelDialog, DetailView, GridView
# from main.tab_view.t_preprocess_view import PreprocessAugmentTab
# from main.tab_view.t_camera_view import CameraAnnotation





class Main(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize settings manager. 
        self.settings_manager = SettingsManager()
        self.custom_label_manager = CustomLabelManager(self.settings_manager)

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

        # Class selection mode: "model_only", "custom_only", or "both"
        self.class_selection_mode = "both"  # Default to both

        # Load settings or show settings dialog
        if not self.load_settings():
            self.show_settings_dialog()
        else:
            self.initialize_ui()
            self.initialize_auto_suggest()

    def reset_manual_annotation_fnc(self):
        self.detail_view.annotate_qlabel.clear_boxes()

    def manual_annotation_fnc(self):
        try:
            self.detail_view.manual_annotation_enabled = self.detail_view.manual_annotation_btn.isChecked()
            if self.detail_view.manual_annotation_enabled:
                self.detail_view.manual_annotation_btn.setText("Manual Annotation: ON")
                self.detail_view.save_manual_annotation_btn.setDisabled(False)
                self.detail_view.annotate_qlabel.mousePressEvent = self.detail_view.annotate_qlabel.mousePressEvents
                self.detail_view.annotate_qlabel.enable_drawing()
                print("🟢 Annotation mode started.")
            else:
                self.detail_view.manual_annotation_btn.setText("Manual Annotation: OFF")
                self.detail_view.save_manual_annotation_btn.setDisabled(True)
                self.detail_view.annotate_qlabel.mousePressEvent = self.handle_click
                self.detail_view.annotate_qlabel.disable_drawing()
                self.reset_manual_annotation_fnc()
        except Exception as e:
            print("Error on Manual Annotation Image Function: ", e)

    def setup_auto_suggest_combo(self):
        """
        Set up auto-suggest functionality for the manage_label_btn combo box.
        This function should be called after initializing the combo box.
        """
        # Create a completer for auto-suggestions
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)  # Match anywhere in the text

        # Set the completer to the combo box
        self.detail_view.label_combo.setCompleter(self.completer)

        # Update the completer model when combo box items change
        self.update_completer_model()

    def update_completer_model(self):
        """
        Update the completer model with current combo box items.
        Call this whenever you add new items to the combo box.
        """
        if not hasattr(self, 'completer'):
            return

        # Get all items from combo box
        items = []
        for i in range(self.detail_view.label_combo.count()):
            items.append(self.detail_view.label_combo.itemText(i))

        # Create a model for the completer
        model = QStandardItemModel()
        for item in items:
            model.appendRow(QStandardItem(item))

        # Set the model to the completer
        self.completer.setModel(model)

    def populate_label_combo_with_autosuggest(self):
        """
        Enhanced version of populate_label_combo that includes auto-suggest updates.
        Replace your existing populate_label_combo method with this one.
        """
        self.detail_view.label_combo.clear()

        # Add labels based on the selected mode
        if self.class_selection_mode == "model_only":
            # Add only model classes
            if self.class_names:
                for class_name in self.class_names.values():
                    self.detail_view.label_combo.addItem(class_name)
        elif self.class_selection_mode == "custom_only":
            # Add only custom labels
            for label_name in self.custom_label_manager.get_custom_labels_list():
                self.detail_view.label_combo.addItem(label_name)
        else:  # "both" mode
            # Add model classes
            if self.class_names:
                for class_name in self.class_names.values():
                    self.detail_view.label_combo.addItem(class_name)
            # Add custom labels
            for label_name in self.custom_label_manager.get_custom_labels_list():
                self.detail_view.label_combo.addItem(label_name)

        # Update the auto-suggest completer
        self.update_completer_model()

    # Additional method to add custom suggestions on-the-fly
    def add_suggestion_to_combo(self, suggestion):
        """
        Add a new suggestion to the combo box and update auto-suggest.

        Args:
            suggestion (str): The new suggestion to add
        """
        # Check if item already exists
        existing_items = [self.detail_view.label_combo.itemText(i) for i in range(self.detail_view.label_combo.count())]

        if suggestion not in existing_items:
            # Only add if it's allowed by the current class selection mode
            if self.is_label_allowed(suggestion):
                self.detail_view.label_combo.addItem(suggestion)
                self.update_completer_model()

    def is_label_allowed(self, label_text):
        """
        Check if a manage_label_btn is allowed based on the current class selection mode.

        Args:
            label_text (str): The manage_label_btn text to check

        Returns:
            bool: True if the manage_label_btn is allowed, False otherwise
        """
        if self.class_selection_mode == "model_only":
            # Only allow model class names
            return label_text in self.class_names.values() if self.class_names else False
        elif self.class_selection_mode == "custom_only":
            # Allow any manage_label_btn (will become custom)
            return True
        else:  # "both" mode
            # Allow any manage_label_btn
            return True

    # Enhanced assign manage_label_btn function that learns from user input
    def assign_label_to_selected_with_learning(self):
        """
        Enhanced version of assign_label_to_selected that learns new labels for auto-suggest.
        """
        if not self.selected_boxes:
            QMessageBox.warning(self, "Warning", "No boxes selected.")
            return

        label_text = self.detail_view.label_combo.currentText().strip()
        if not label_text:
            QMessageBox.warning(self, "Warning", "Please enter or select a manage_label_btn.")
            return

        # Check if the manage_label_btn is allowed based on current mode
        if not self.is_label_allowed(label_text):
            mode_text = {
                "model_only": "model classes only",
                "custom_only": "custom labels only",
                "both": "both model and custom labels"
            }
            QMessageBox.warning(self, "Label Not Allowed",
                                f"The manage_label_btn '{label_text}' is not allowed in the current mode: {mode_text[self.class_selection_mode]}.")
            return

        # Add the new manage_label_btn to suggestions if it's not already there
        self.add_suggestion_to_combo(label_text)

        try:
            # Get or create class ID for this manage_label_btn with current mode
            class_id = self.custom_label_manager.get_class_id(
                label_text,
                self.class_names,
                self.class_selection_mode
            )

            # Assign manage_label_btn to all selected boxes
            for i in self.selected_boxes:
                self.box_labels[i] = label_text
                # Update the detection's class ID
                self.detections[i]["cls"] = class_id

            # Refresh combo box to include any new custom labels
            self.populate_label_combo_with_autosuggest()

            # Set current text back to the assigned manage_label_btn
            self.detail_view.label_combo.setCurrentText(label_text)

            self.has_unsaved_changes = True
            self.display_image()
            if self.detail_view.auto_save_enabled:
                self.save_annotations()

        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    # Method to initialize auto-suggest in your existing setup
    def initialize_auto_suggest(self):
        """
        Initialize auto-suggest functionality.
        Call this in your initialize_ui method after setting up the combo box.
        """
        # Set up auto-suggest
        self.setup_auto_suggest_combo()

        # Populate combo box with auto-suggest
        self.populate_label_combo_with_autosuggest()

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

            # Load class selection mode from settings
            self.class_selection_mode = settings.get('class_selection_mode', 'both')

            # Ensure directories exist
            os.makedirs(self.label_dir, exist_ok=True)
            os.makedirs(self.annotation_save_dir, exist_ok=True)
            self.class_names = self.model.names
            self.custom_label_manager.set_model_classes(self.class_names)

            # Set the mode in the custom manage_label_btn manager
            self.custom_label_manager.set_mode(self.class_selection_mode)

            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading settings: {e}")
            return False

    def show_settings_dialog(self):
        """Show settings configuration dialog"""
        current_settings = self.settings_manager.load_settings()
        current_settings['class_selection_mode'] = getattr(self, 'class_selection_mode', 'both')
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


    def check_albumentations_dependency(self):
        """Check if albumentations is installed"""
        try:
            import albumentations
            return True
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                                 "The albumentations library is required for preprocessing and augmentation features.\n\n"
                                 "Please install it using:\n"
                                 "pip install albumentations\n\n"
                                 "The Preprocess & Augment tab will be disabled until this dependency is installed.")
            return False

    def initialize_ui(self):
        """Initialize the main UI after settings are loaded"""
        self.setWindowTitle("YOLOv11 Auto Annotate GUI - Enhanced with Grid View and Augmentation")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_ui()

        # Check for albumentations dependency
        if not self.check_albumentations_dependency():
            # Disable the augmentation tab
            self.tab_widget.setTabEnabled(2, False)
            self.tab_widget.setTabToolTip(2, "Install albumentations library to enable this feature")

        self.connect_signals()
        self.show()


    def populate_label_combo(self):
        """Populate combo box with available labels based on selection mode"""
        self.detail_view.label_combo.clear()

        if self.class_selection_mode == "model_only":
            # Add only model classes
            if self.class_names:
                for class_name in self.class_names.values():
                    self.detail_view.label_combo.addItem(class_name)
        elif self.class_selection_mode == "custom_only":
            # Add only custom labels
            for label_name in self.custom_label_manager.get_custom_labels_list():
                self.detail_view.label_combo.addItem(label_name)
        else:  # "both" mode
            # Add model classes
            if self.class_names:
                for class_name in self.class_names.values():
                    self.detail_view.label_combo.addItem(class_name)
            # Add custom labels
            for label_name in self.custom_label_manager.get_custom_labels_list():
                self.detail_view.label_combo.addItem(label_name)


    def setup_settings_button(self, parent_layout):
        settings_layout = QHBoxLayout()
        self.settings_btn = QPushButton("Settings")
        settings_layout.addWidget(self.settings_btn)
        settings_layout.addStretch()
        parent_layout.addLayout(settings_layout)


    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Add settings button at the top
        self.setup_settings_button(main_layout)

        # Create tab widget for grid and detail views
        self.tab_widget = QTabWidget()

        # Grid View Tab
        self.grid_view = GridView(self)
        self.tab_widget.addTab(self.grid_view, "Grid view")


        # Detail View Tab
        self.detail_view = DetailView(self)
        self.tab_widget.addTab(self.detail_view, "Detail view")

        # Preprocess and Augment Tab
        self.preprocess_augment_tab = PreprocessView(self)
        self.tab_widget.addTab(self.preprocess_augment_tab, "Preprocess & Augment")

        # Camera Annotation Tab
        self.camera_annotation = CameraAnnotation(self)
        self.tab_widget.addTab(self.camera_annotation, "Camera Capture Annotation")
        self.tab_widget.currentChanged.connect(self.auto_save_current_annotation)


        # Set default selection
        if self.class_selection_mode == "model_only":
            self.detail_view.model_only_radio.setChecked(True)
        elif self.class_selection_mode == "custom_only":
            self.detail_view.custom_only_radio.setChecked(True)
        else:
            self.detail_view.both_radio.setChecked(True)

        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)


    def on_class_mode_changed(self):
        """Handle class selection mode change"""
        # old_mode = self.class_selection_mode

        if self.detail_view.model_only_radio.isChecked():
            self.class_selection_mode = "model_only"
        elif self.detail_view.custom_only_radio.isChecked():
            self.class_selection_mode = "custom_only"
        else:
            self.class_selection_mode = "both"

        # Update the custom manage_label_btn manager mode
        self.custom_label_manager.set_mode(self.class_selection_mode)

        # Update the combo box with new mode
        self.populate_label_combo_with_autosuggest()

        # Update placeholder text based on mode
        if self.class_selection_mode == "model_only":
            self.detail_view.label_combo.setPlaceholderText("Select model class")
        elif self.class_selection_mode == "custom_only":
            self.detail_view.label_combo.setPlaceholderText("Enter custom manage_label_btn")
        else:
            self.detail_view.label_combo.setPlaceholderText("Select or enter manage_label_btn")


    def connect_signals(self):
        # self.load_btn.clicked.connect(self.load_images)
        self.settings_btn.clicked.connect(self.show_settings_dialog)
        self.grid_view.select_all_thumbnails_btn.clicked.connect(self.select_all_thumbnails)
        self.grid_view.deselect_all_thumbnails_btn.clicked.connect(self.deselect_all_thumbnails)
        self.grid_view.remove_selected_btn.clicked.connect(self.remove_selected_images)
        self.grid_view.add_images_btn.clicked.connect(self.add_images)
        self.grid_view.remove_all_btn.clicked.connect(self.remove_all_images)
        self.grid_view.export_annotations_btn.clicked.connect(self.export_annotations)
        self.detail_view.next_btn.clicked.connect(self.show_next_image)
        self.detail_view.prev_btn.clicked.connect(self.show_prev_image)
        self.detail_view.save_btn.clicked.connect(self.save_annotations)
        self.detail_view.manual_annotation_btn.clicked.connect(self.manual_annotation_fnc)
        self.detail_view.reset_manual_annotation_btn.clicked.connect(self.reset_manual_annotation_fnc)
        self.detail_view.save_manual_annotation_btn.clicked.connect(self.save_manual_annotation_fnc)
        self.detail_view.assign_label_btn.clicked.connect(self.assign_label_to_selected_with_learning)
        self.detail_view.select_all_btn.clicked.connect(self.select_all_boxes)
        self.detail_view.deselect_all_btn.clicked.connect(self.deselect_all_boxes)
        self.detail_view.remove_selected_annotation_btn.clicked.connect(self.remove_annotation)
        self.detail_view.auto_save_btn.clicked.connect(self.toggle_auto_save)
        self.detail_view.clear_session_btn.clicked.connect(self.clear_all_annotations)
        self.detail_view.annotate_qlabel.mousePressEvent = self.handle_click
        self.detail_view.manage_labels_btn.clicked.connect(self.show_label_management_dialog)
        self.detail_view.refresh_labels_btn.clicked.connect(self.populate_label_combo_with_autosuggest)
        self.detail_view.class_mode_group.buttonClicked.connect(self.on_class_mode_changed)

    def show_label_management_dialog(self):
        """Show the custom manage_label_btn management dialog"""
        dialog = CustomLabelDialog(self.custom_label_manager, self)
        if dialog.exec_() == QDialog.Accepted:
            # Refresh the combo box after dialog closes
            self.populate_label_combo_with_autosuggest()

    def on_thumbnail_clicked(self, index):
        # Auto-save current annotation before switching if enabled
        if self.detail_view.auto_save_enabled and self.image_paths and self.detections and hasattr(self, 'current_index'):
            self.auto_save_current_annotation()

        self.current_index = index
        self.show_current_image()
        # Switch to detail view
        self.tab_widget.setCurrentIndex(1)
        # Update thumbnail states
        self.update_thumbnail_states()

    def save_manual_annotation_fnc(self):
        try:
            if not self.detail_view.label_combo.currentText().strip() == "":
                if not hasattr(self, 'current_image'):
                    print("No current image loaded")
                    return
                index = self.detail_view.label_combo.currentIndex()
                # Get original image dimensions
                original_height, original_width = self.current_image.shape[:2]

                # Get the displayed pixmap and its dimensions
                pixmap = self.detail_view.annotate_qlabel.pixmap()
                if pixmap is None:
                    print("No pixmap in image_label")
                    return

                displayed_width = pixmap.width()
                displayed_height = pixmap.height()

                # Get the manage_label_btn size
                label_width = self.detail_view.annotate_qlabel.width()
                label_height = self.detail_view.annotate_qlabel.height()

                # Calculate the actual position of the scaled image within the manage_label_btn
                # (since the image is centered when scaled with KeepAspectRatio)
                x_offset = (label_width - displayed_width) // 2
                y_offset = (label_height - displayed_height) // 2

                # Calculate scaling factors
                scale_x = original_width / displayed_width
                scale_y = original_height / displayed_height

                # Convert manual annotations to original image coordinates
                for idx, rect in enumerate(self.detail_view.annotate_qlabel.boxes):
                    # Get coordinates relative to the manage_label_btn
                    label_x1, label_y1 = rect.left(), rect.top()
                    label_x2, label_y2 = rect.right(), rect.bottom()

                    # Convert to coordinates relative to the displayed image
                    img_x1 = label_x1 - x_offset
                    img_y1 = label_y1 - y_offset
                    img_x2 = label_x2 - x_offset
                    img_y2 = label_y2 - y_offset

                    # Skip boxes that are outside the image area
                    if img_x1 < 0 or img_y1 < 0 or img_x2 > displayed_width or img_y2 > displayed_height:
                        continue

                    # Scale to original image coordinates
                    orig_x1 = int(img_x1 * scale_x)
                    orig_y1 = int(img_y1 * scale_y)
                    orig_x2 = int(img_x2 * scale_x)
                    orig_y2 = int(img_y2 * scale_y)

                    # Clamp to image boundaries
                    orig_x1 = max(0, min(orig_x1, original_width))
                    orig_y1 = max(0, min(orig_y1, original_height))
                    orig_x2 = max(0, min(orig_x2, original_width))
                    orig_y2 = max(0, min(orig_y2, original_height))

                    # Calculate area for sorting
                    area = (orig_x2 - orig_x1) * (orig_y2 - orig_y1)

                    # Only add if the box has some area
                    if area > 0:
                        self.detections.append({
                            "cls": index,  # Default class, you might want to make this configurable
                            "conf": 1.0,
                            "box": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "area": area,
                            "original_index": len(self.detections)  # Use actual index
                        })

                # Clear the manual annotation boxes
                self.detail_view.annotate_qlabel.clear_boxes()

                # Update UI state
                self.detail_view.manual_annotation_btn.setChecked(False)
                self.detail_view.manual_annotation_btn.setText("Manual Annotation: OFF")
                self.detail_view.save_manual_annotation_btn.setDisabled(True)

                # Restore normal mouse handling
                self.detail_view.annotate_qlabel.mousePressEvent = self.handle_click
                self.detail_view.annotate_qlabel.disable_drawing()
                self.detections.sort(key=lambda x: x["area"], reverse=True)
                # Refresh display and status
                self.display_image()
                self.update_thumbnail_states()
                self.update_status()

                print(f"Added {len(self.detail_view.annotate_qlabel.boxes)} manual annotations")
            else:
                QMessageBox.information(self, "Info", "Annotation manage_label_btn is empty!")
        except Exception as e:
            print("Error in save_manual_annotation_fnc:", e)
            import traceback
            traceback.print_exc()


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
        if self.detail_view.auto_save_enabled and self.image_paths and self.detections:
            self.auto_save_current_annotation()

        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_thumbnail_states()
        else:
            QMessageBox.information(self, "Info", "This is the last image.")

    def show_prev_image(self):
        # Auto-save current annotation before moving if enabled
        if self.detail_view.auto_save_enabled and self.image_paths and self.detections:
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
        # Get labels based on current mode
        all_labels = self.custom_label_manager.get_all_labels(
            self.class_names,
            self.class_selection_mode
        )

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

            # Display manage_label_btn (custom or original class name)
            if i in self.box_labels:
                label_text = self.box_labels[i]
            else:
                # Use the comprehensive manage_label_btn mapping based on current mode
                label_text = all_labels.get(cls, f"Unknown_{cls}")

            # Add confidence score
            label_text += f" ({det['conf']:.2f})"

            cv2.putText(img, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image)

        # Scale image to fit manage_label_btn while maintaining aspect ratio
        scaled_pix = pix.scaled(self.detail_view.annotate_qlabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.detail_view.annotate_qlabel.setPixmap(scaled_pix)



    def handle_click(self, event):
        if not self.detections:
            return

        x = event.pos().x()
        y = event.pos().y()
        label_w, label_h = self.detail_view.annotate_qlabel.width(), self.detail_view.annotate_qlabel.height()
        img_h, img_w = self.original_shape

        # Calculate resize ratio
        pixmap = self.detail_view.annotate_qlabel.pixmap()
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

        label_text = self.detail_view.label_combo.currentText().strip()
        if not label_text:
            QMessageBox.warning(self, "Warning", "Please enter or select a manage_label_btn.")
            return

        # Check if the manage_label_btn is allowed based on current mode
        if not self.is_label_allowed(label_text):
            mode_text = {
                "model_only": "model classes only",
                "custom_only": "custom labels only",
                "both": "both model and custom labels"
            }
            QMessageBox.warning(self, "Label Not Allowed",
                                f"The manage_label_btn '{label_text}' is not allowed in the current mode: {mode_text[self.class_selection_mode]}.")
            return

        # Get or create class ID for this manage_label_btn
        class_id = self.custom_label_manager.get_class_id(label_text, self.class_names)

        # Assign manage_label_btn to all selected boxes
        for i in self.selected_boxes:
            self.box_labels[i] = label_text
            # Update the detection's class ID
            self.detections[i]["cls"] = class_id

        # Refresh combo box to include any new custom labels
        self.populate_label_combo_with_autosuggest()

        # Set current text back to the assigned manage_label_btn
        self.detail_view.label_combo.setCurrentText(label_text)

        self.has_unsaved_changes = True
        self.display_image()
        if self.detail_view.auto_save_enabled:
            self.save_annotations()

    def remove_annotation(self):
        print("Removed!")
        for i in sorted(self.selected_boxes, reverse=True):
            if 0 <= i < len(self.detections):
                del self.detections[i]

        # Clear selected_boxes after deletion
        self.selected_boxes.clear()
        self.display_image()
        self.update_status()

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
                status_text += "(Modified)"
            elif self.has_saved_annotation(self.current_index):
                status_text += "(Saved)"
            self.detail_view.status_label.setText(status_text)
        self.detail_view.selection_count_label.setText(f"Selected: {len(self.selected_boxes)} / {len(self.detections)}")

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

    def save_yolo_format(self):
        """Save annotations in YOLO format with consistent class IDs"""
        img_path = self.image_paths[self.current_index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        h, w = self.original_shape

        label_file = os.path.join(self.label_dir, f"{img_name}.txt")
        with open(label_file, "w") as f:
            for i in self.selected_boxes:
                det = self.detections[i]
                x1, y1, x2, y2 = det["box"]

                # Get class ID using the custom manage_label_btn manager with current mode
                if i in self.box_labels:
                    custom_label = self.box_labels[i]
                    cls = self.custom_label_manager.get_class_id(
                        custom_label,
                        self.class_names,
                        self.class_selection_mode
                    )
                else:
                    cls = det["cls"]

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
            "timestamp": str(np.datetime64('now')),
            # "custom_labels": self.custom_label_manager.custom_labels,  # Save custom labels mapping
            "class_selection_mode": self.class_selection_mode  # Save current mode
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

            # # Verify the image path matches
            # if annotation_data["image_path"] != self.image_paths[self.current_index]:
            #     return False

            self.detections = annotation_data["detections"]
            self.selected_boxes = set(annotation_data["selected_boxes"])
            self.box_labels = {int(k): v for k, v in annotation_data["box_labels"].items()}

            # # Load custom labels if available in the annotation
            # if "custom_labels" in annotation_data:
            #     # Merge with existing custom labels
            #     for label_name, class_id in annotation_data["custom_labels"].items():
            #         if label_name not in self.custom_label_manager.custom_labels:
            #             self.custom_label_manager.custom_labels[label_name] = class_id
            #             # Update next_custom_id if necessary
            #             if class_id >= self.custom_label_manager.next_custom_id:
            #                 self.custom_label_manager.next_custom_id = class_id + 1
            #
            #     # Save the updated custom labels
            #     self.custom_label_manager.save_custom_labels()
            #     # Refresh combo box
            #     self.populate_label_combo_with_autosuggest()

            return True

        except Exception as e:
            print(f"Error loading annotation: {e}")
            return False

    def toggle_auto_save(self):
        """Toggle auto-save functionality"""
        self.detail_view.auto_save_enabled = self.detail_view.auto_save_btn.isChecked()
        if self.detail_view.auto_save_enabled:
            self.detail_view.auto_save_btn.setText("Auto Save: ON")
            # Auto-save current state if there are unsaved changes
            if self.has_unsaved_changes:
                self.auto_save_current_annotation()
        else:
            self.detail_view.auto_save_btn.setText("Auto Save: OFF")

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
                if self.detail_view.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
                    self.auto_save_current_annotation()

                # Clear all thumbnails from the grid
                for thumbnail in self.thumbnails:
                    thumbnail.deleteLater()
                self.thumbnails.clear()

                # Clear the grid layout
                for i in reversed(range(self.grid_view.grid_layout.count())):
                    child = self.grid_view.grid_layout.itemAt(i).widget()
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
                self.detail_view.annotate_qlabel.clear()
                self.detail_view.annotate_qlabel.setText("Load images to start")

                # Clear the manage_label_btn combo box (keep only model class names based on mode)
                self.populate_label_combo_with_autosuggest()

                # Update status labels
                self.detail_view.status_label.setText("Load images to start")
                self.detail_view.selection_count_label.setText("Selected: 0")

                # Switch to grid view tab
                self.tab_widget.setCurrentIndex(0)

                QMessageBox.information(self, "Complete", "All images have been removed from the session.")
                self.update_thumbnail_selection_ui()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing images: {e}")

    def remove_loaded_images(self):
        """Remove all loaded images from the thumbnail grid and reset the application state"""
        if not self.image_paths:
            QMessageBox.information(self, "Info", "No images are currently loaded.")
            return

        try:
            # Auto-save current annotation before clearing if enabled and there are unsaved changes
            if self.detail_view.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
                self.auto_save_current_annotation()

            # Clear all thumbnails from the grid
            for thumbnail in self.thumbnails:
                thumbnail.deleteLater()
            self.thumbnails.clear()

            # Clear the grid layout
            for i in reversed(range(self.grid_view.grid_layout.count())):
                child = self.grid_view.grid_layout.itemAt(i).widget()
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
            self.detail_view.annotate_qlabel.clear()
            self.detail_view.annotate_qlabel.setText("Load images to start")

            # Clear the manage_label_btn combo box based on current mode
            self.populate_label_combo_with_autosuggest()

            # Update status labels
            self.detail_view.status_label.setText("Load images to start")
            self.detail_view.selection_count_label.setText("Selected: 0")

            # Switch to grid view tab
            self.tab_widget.setCurrentIndex(0)

            QMessageBox.information(self, "Complete", "All images have been removed from the session.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing images: {e}")

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
            images_dir = os.path.join(export_dir, "../images")
            labels_dir = os.path.join(export_dir, "../labels")
            annotations_dir = os.path.join(export_dir, "../annotations")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)

            exported_count = 0
            failed_count = 0

            # Create class mapping file for custom labels
            all_labels = self.custom_label_manager.get_all_labels(self.class_names)
            classes_file = os.path.join(export_dir, "classes.txt")
            with open(classes_file, "w") as f:
                # Sort by class ID for consistent ordering
                sorted_labels = sorted(all_labels.items())
                for class_id, class_name in sorted_labels:
                    f.write(f"{class_name}\n")

            # Export custom labels mapping
            if self.custom_label_manager.custom_labels:
                custom_labels_export = os.path.join(export_dir, "custom_labels_mapping.json")
                with open(custom_labels_export, "w") as f:
                    json.dump(self.custom_label_manager.custom_labels, f, indent=2)

            for i in range(len(self.image_paths)):
                if self.has_saved_annotation(i):
                    try:
                        img_path = self.image_paths[i]
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        img_ext = os.path.splitext(os.path.basename(img_path))[1]

                        # Move the original image
                        dest_img_path = os.path.join(images_dir, f"{img_name}{img_ext}")
                        shutil.move(img_path, dest_img_path)

                        # Move manage_label_btn file
                        label_file = os.path.join(self.label_dir, f"{img_name}.txt")
                        if os.path.exists(label_file):
                            dest_label_path = os.path.join(labels_dir, f"{img_name}.txt")
                            shutil.move(label_file, dest_label_path)

                        # Move annotation file
                        annotation_file = self.get_annotation_filename(i)
                        if os.path.exists(annotation_file):
                            dest_annotation_path = os.path.join(annotations_dir, f"{img_name}_annotation.json")
                            shutil.move(annotation_file, dest_annotation_path)

                        exported_count += 1

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
                f.write(f"Total Annotated Images Found: {annotated_count}\n")
                f.write(f"Custom Labels Used: {len(self.custom_label_manager.custom_labels)}\n")
                f.write(f"Class Selection Mode: {self.class_selection_mode}\n\n")
                f.write(f"Directory Structure:\n")
                f.write(f"- images/: Contains the original image files\n")
                f.write(f"- labels/: Contains YOLO format manage_label_btn files (.txt)\n")
                f.write(f"- annotations/: Contains detailed annotation files (.json)\n")
                f.write(f"- classes.txt: Contains all class names in order\n")
                f.write(f"- custom_labels_mapping.json: Maps custom labels to class IDs\n")

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
        if self.detail_view.auto_save_enabled and self.image_paths and self.detections and self.has_unsaved_changes:
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
                self.custom_label_manager.set_model_classes(self.class_names)
                self.populate_label_combo_with_autosuggest()
            else:
                # Just refresh the combo box based on current mode
                self.populate_label_combo_with_autosuggest()

            # Create thumbnails for the new images only
            self.add_new_thumbnails(new_files, old_count)

            # Update status
            new_count = len(new_files)
            total_count = len(self.image_paths)

            status_message = f"Added {new_count} new images. Total: {total_count} images"
            if existing_files:
                status_message += f" ({len(existing_files)} duplicates skipped)"

            self.detail_view.status_label.setText(status_message)

            # Show info message
            info_message = f"Successfully added {new_count} new images."
            if existing_files:
                info_message += f"\n{len(existing_files)} duplicate(s) were skipped."
            self.update_thumbnail_selection_ui()
            QMessageBox.information(self, "Images Added", info_message)

    def add_new_thumbnails(self, new_files, start_index):
        """Create thumbnails for newly added images and add them to the grid"""
        cols = 5

        for i, img_path in enumerate(new_files):
            thumbnail_index = start_index + i
            thumbnail = ImageThumbnail(img_path, thumbnail_index)
            thumbnail.clicked.connect(self.on_thumbnail_clicked)
            thumbnail.selection_changed.connect(self.update_thumbnail_selection_ui)

            # Calculate grid position
            row = thumbnail_index // cols
            col = thumbnail_index % cols
            self.grid_view.grid_layout.addWidget(thumbnail, row, col)
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
                if (self.detail_view.auto_save_enabled and self.current_index in selected_indices and
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
                for i in reversed(range(self.grid_view.grid_layout.count())):
                    child = self.grid_view.grid_layout.itemAt(i).widget()
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
                        self.grid_view.grid_layout.addWidget(thumbnail, row, col)

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
                    self.detail_view.annotate_qlabel.clear()
                    self.detail_view.annotate_qlabel.setText("Load images to start")

                    # Update status labels
                    self.detail_view.status_label.setText("Load images to start")
                    self.detail_view.selection_count_label.setText("Selected: 0")

                QMessageBox.information(self, "Complete",
                                        f"Successfully removed {removed_count} image(s). "
                                        f"Remaining images: {len(self.image_paths)}")
                self.update_thumbnail_selection_ui()
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
        if hasattr(self.grid_view, 'thumbnail_selection_label'):
            self.grid_view.thumbnail_selection_label.setText(f"Selected: {selected_count} / {len(self.thumbnails)}")

    def get_class_selection_summary(self):
        """Get a summary of the current class selection mode for display"""
        mode_descriptions = {
            "model_only": "Using model classes only",
            "custom_only": "Using custom labels only",
            "both": "Using both model classes and custom labels"
        }
        return mode_descriptions.get(self.class_selection_mode, "Unknown mode")

    def validate_label_for_mode(self, label_text):
        """
        Validate if a manage_label_btn is appropriate for the current selection mode.

        Args:
            label_text (str): The manage_label_btn to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        if self.class_selection_mode == "model_only":
            if self.class_names and label_text not in self.class_names.values():
                available_classes = ", ".join(self.class_names.values())
                return False, f"Only model classes are allowed. Available: {available_classes}"
        elif self.class_selection_mode == "custom_only":
            if self.class_names and label_text in self.class_names.values():
                return False, "Only custom labels are allowed. This is a model class."
        # "both" mode allows everything
        return True, ""

    def update_ui_for_mode_change(self):
        """Update UI elements when class selection mode changes"""
        # Update combo box placeholder
        placeholders = {
            "model_only": "Select model class",
            "custom_only": "Enter custom manage_label_btn",
            "both": "Select or enter manage_label_btn"
        }
        self.detail_view.label_combo.setPlaceholderText(placeholders.get(self.class_selection_mode, "Select or enter manage_label_btn"))

        # Update status display if needed
        if hasattr(self, 'status_label'):
            current_text = self.detail_view.status_label.text()
            if "Mode:" not in current_text and self.image_paths:
                mode_text = self.get_class_selection_summary()
                self.detail_view.status_label.setText(f"{current_text} | Mode: {self.class_selection_mode}")

    def export_class_configuration(self, export_dir):
        """Export class configuration information"""
        try:
            config_file = os.path.join(export_dir, "class_configuration.json")
            config_data = {
                "class_selection_mode": self.class_selection_mode,
                "model_classes": dict(self.class_names) if self.class_names else {},
                "custom_labels": self.custom_label_manager.custom_labels,
                "total_model_classes": len(self.class_names) if self.class_names else 0,
                "total_custom_labels": len(self.custom_label_manager.custom_labels),
                "export_timestamp": str(np.datetime64('now'))
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error exporting class configuration: {e}")
            return False