import cv2
import os
import json
import albumentations as A
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSlider, QSpinBox, QCheckBox, QComboBox,
    QProgressBar, QTextEdit, QScrollArea, QGridLayout,
    QMessageBox
)
from PyQt5.QtCore import Qt
from .augment_worker import AugmentationWorker


class PreprocessView(QWidget):
    def __init__(self, parent_annotator):
        super().__init__()
        self.parent_annotator = parent_annotator
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        # Create scroll area for the entire tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # Preprocessing Section
        preprocess_group = self.create_preprocessing_section()
        scroll_layout.addWidget(preprocess_group)

        # Augmentation Section
        augmentation_group = self.create_augmentation_section()
        scroll_layout.addWidget(augmentation_group)

        # Output Section
        output_group = self.create_output_section()
        scroll_layout.addWidget(output_group)

        # Control buttons
        control_group = self.create_control_section()
        scroll_layout.addWidget(control_group)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)

    def create_preprocessing_section(self):
        """Create preprocessing controls"""
        group = QGroupBox("Preprocessing")
        layout = QVBoxLayout()

        # Resize options
        resize_layout = QHBoxLayout()
        self.resize_checkbox = QCheckBox("Resize Images")
        self.resize_width = QSpinBox()
        self.resize_width.setRange(32, 2048)
        self.resize_width.setValue(640)
        self.resize_height = QSpinBox()
        self.resize_height.setRange(32, 2048)
        self.resize_height.setValue(640)
        self.maintain_aspect = QCheckBox("Maintain Aspect Ratio")
        self.maintain_aspect.setChecked(True)

        resize_layout.addWidget(self.resize_checkbox)
        resize_layout.addWidget(QLabel("Width:"))
        resize_layout.addWidget(self.resize_width)
        resize_layout.addWidget(QLabel("Height:"))
        resize_layout.addWidget(self.resize_height)
        resize_layout.addWidget(self.maintain_aspect)
        resize_layout.addStretch()
        layout.addLayout(resize_layout)

        # Normalization
        norm_layout = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("Normalize (0-1)")
        self.standardize_checkbox = QCheckBox("Standardize (ImageNet)")
        norm_layout.addWidget(self.normalize_checkbox)
        norm_layout.addWidget(self.standardize_checkbox)
        norm_layout.addStretch()
        layout.addLayout(norm_layout)

        # Grayscale conversion
        gray_layout = QHBoxLayout()
        self.grayscale_checkbox = QCheckBox("Convert to Grayscale")
        gray_layout.addWidget(self.grayscale_checkbox)
        gray_layout.addStretch()
        layout.addLayout(gray_layout)

        group.setLayout(layout)
        return group

    def create_augmentation_section(self):
        """Create augmentation controls"""
        group = QGroupBox("Augmentations")
        layout = QVBoxLayout()

        # Number of augmented versions
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Augmented versions per image:"))
        self.aug_count_spin = QSpinBox()
        self.aug_count_spin.setRange(1, 20)
        self.aug_count_spin.setValue(3)
        count_layout.addWidget(self.aug_count_spin)
        count_layout.addStretch()
        layout.addLayout(count_layout)

        # Create augmentation grid
        aug_grid = QGridLayout()

        # Rotation
        self.rotation_checkbox = QCheckBox("Rotation")
        self.rotation_angle = QSlider(Qt.Horizontal)
        self.rotation_angle.setRange(0, 180)
        self.rotation_angle.setValue(15)
        self.rotation_label = QLabel("15°")
        self.rotation_angle.valueChanged.connect(lambda v: self.rotation_label.setText(f"{v}°"))

        aug_grid.addWidget(self.rotation_checkbox, 0, 0)
        aug_grid.addWidget(QLabel("Max Angle:"), 0, 1)
        aug_grid.addWidget(self.rotation_angle, 0, 2)
        aug_grid.addWidget(self.rotation_label, 0, 3)

        # Horizontal Flip
        self.hflip_checkbox = QCheckBox("Horizontal Flip")
        self.hflip_prob = QSlider(Qt.Horizontal)
        self.hflip_prob.setRange(0, 100)
        self.hflip_prob.setValue(50)
        self.hflip_label = QLabel("50%")
        self.hflip_prob.valueChanged.connect(lambda v: self.hflip_label.setText(f"{v}%"))

        aug_grid.addWidget(self.hflip_checkbox, 1, 0)
        aug_grid.addWidget(QLabel("Probability:"), 1, 1)
        aug_grid.addWidget(self.hflip_prob, 1, 2)
        aug_grid.addWidget(self.hflip_label, 1, 3)

        # Vertical Flip
        self.vflip_checkbox = QCheckBox("Vertical Flip")
        self.vflip_prob = QSlider(Qt.Horizontal)
        self.vflip_prob.setRange(0, 100)
        self.vflip_prob.setValue(20)
        self.vflip_label = QLabel("20%")
        self.vflip_prob.valueChanged.connect(lambda v: self.vflip_label.setText(f"{v}%"))

        aug_grid.addWidget(self.vflip_checkbox, 2, 0)
        aug_grid.addWidget(QLabel("Probability:"), 2, 1)
        aug_grid.addWidget(self.vflip_prob, 2, 2)
        aug_grid.addWidget(self.vflip_label, 2, 3)

        # Brightness
        self.brightness_checkbox = QCheckBox("Brightness")
        self.brightness_limit = QSlider(Qt.Horizontal)
        self.brightness_limit.setRange(0, 50)
        self.brightness_limit.setValue(20)
        self.brightness_label = QLabel("20%")
        self.brightness_limit.valueChanged.connect(lambda v: self.brightness_label.setText(f"{v}%"))

        aug_grid.addWidget(self.brightness_checkbox, 3, 0)
        aug_grid.addWidget(QLabel("Limit:"), 3, 1)
        aug_grid.addWidget(self.brightness_limit, 3, 2)
        aug_grid.addWidget(self.brightness_label, 3, 3)

        # Contrast
        self.contrast_checkbox = QCheckBox("Contrast")
        self.contrast_limit = QSlider(Qt.Horizontal)
        self.contrast_limit.setRange(0, 50)
        self.contrast_limit.setValue(20)
        self.contrast_label = QLabel("20%")
        self.contrast_limit.valueChanged.connect(lambda v: self.contrast_label.setText(f"{v}%"))

        aug_grid.addWidget(self.contrast_checkbox, 4, 0)
        aug_grid.addWidget(QLabel("Limit:"), 4, 1)
        aug_grid.addWidget(self.contrast_limit, 4, 2)
        aug_grid.addWidget(self.contrast_label, 4, 3)

        # Saturation
        self.saturation_checkbox = QCheckBox("Saturation")
        self.saturation_limit = QSlider(Qt.Horizontal)
        self.saturation_limit.setRange(0, 50)
        self.saturation_limit.setValue(20)
        self.saturation_label = QLabel("20%")
        self.saturation_limit.valueChanged.connect(lambda v: self.saturation_label.setText(f"{v}%"))

        aug_grid.addWidget(self.saturation_checkbox, 5, 0)
        aug_grid.addWidget(QLabel("Limit:"), 5, 1)
        aug_grid.addWidget(self.saturation_limit, 5, 2)
        aug_grid.addWidget(self.saturation_label, 5, 3)

        # Hue
        self.hue_checkbox = QCheckBox("Hue Shift")
        self.hue_limit = QSlider(Qt.Horizontal)
        self.hue_limit.setRange(0, 50)
        self.hue_limit.setValue(10)
        self.hue_label = QLabel("10%")
        self.hue_limit.valueChanged.connect(lambda v: self.hue_label.setText(f"{v}%"))

        aug_grid.addWidget(self.hue_checkbox, 6, 0)
        aug_grid.addWidget(QLabel("Limit:"), 6, 1)
        aug_grid.addWidget(self.hue_limit, 6, 2)
        aug_grid.addWidget(self.hue_label, 6, 3)

        # Gaussian Blur
        self.blur_checkbox = QCheckBox("Gaussian Blur")
        self.blur_limit = QSlider(Qt.Horizontal)
        self.blur_limit.setRange(1, 10)
        self.blur_limit.setValue(3)
        self.blur_label = QLabel("3px")
        self.blur_limit.valueChanged.connect(lambda v: self.blur_label.setText(f"{v}px"))

        aug_grid.addWidget(self.blur_checkbox, 7, 0)
        aug_grid.addWidget(QLabel("Max Blur:"), 7, 1)
        aug_grid.addWidget(self.blur_limit, 7, 2)
        aug_grid.addWidget(self.blur_label, 7, 3)

        # Gaussian Noise
        self.noise_checkbox = QCheckBox("Gaussian Noise")
        self.noise_limit = QSlider(Qt.Horizontal)
        self.noise_limit.setRange(1, 50)
        self.noise_limit.setValue(10)
        self.noise_label = QLabel("10")
        self.noise_limit.valueChanged.connect(lambda v: self.noise_label.setText(f"{v}"))

        aug_grid.addWidget(self.noise_checkbox, 8, 0)
        aug_grid.addWidget(QLabel("Variance:"), 8, 1)
        aug_grid.addWidget(self.noise_limit, 8, 2)
        aug_grid.addWidget(self.noise_label, 8, 3)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe_checkbox = QCheckBox("CLAHE")
        self.clahe_limit = QSlider(Qt.Horizontal)
        self.clahe_limit.setRange(1, 10)
        self.clahe_limit.setValue(4)
        self.clahe_label = QLabel("4.0")
        self.clahe_limit.valueChanged.connect(lambda v: self.clahe_label.setText(f"{v}.0"))

        aug_grid.addWidget(self.clahe_checkbox, 9, 0)
        aug_grid.addWidget(QLabel("Clip Limit:"), 9, 1)
        aug_grid.addWidget(self.clahe_limit, 9, 2)
        aug_grid.addWidget(self.clahe_label, 9, 3)

        # Random Crop
        self.crop_checkbox = QCheckBox("Random Crop")
        self.crop_scale_min = QSlider(Qt.Horizontal)
        self.crop_scale_min.setRange(50, 95)
        self.crop_scale_min.setValue(80)
        self.crop_label = QLabel("80%")
        self.crop_scale_min.valueChanged.connect(lambda v: self.crop_label.setText(f"{v}%"))

        aug_grid.addWidget(self.crop_checkbox, 10, 0)
        aug_grid.addWidget(QLabel("Min Scale:"), 10, 1)
        aug_grid.addWidget(self.crop_scale_min, 10, 2)
        aug_grid.addWidget(self.crop_label, 10, 3)

        # Cutout/Random Erasing
        self.cutout_checkbox = QCheckBox("Cutout")
        self.cutout_holes = QSlider(Qt.Horizontal)
        self.cutout_holes.setRange(1, 8)
        self.cutout_holes.setValue(2)
        self.cutout_label = QLabel("2 holes")
        self.cutout_holes.valueChanged.connect(lambda v: self.cutout_label.setText(f"{v} holes"))

        aug_grid.addWidget(self.cutout_checkbox, 11, 0)
        aug_grid.addWidget(QLabel("Max Holes:"), 11, 1)
        aug_grid.addWidget(self.cutout_holes, 11, 2)
        aug_grid.addWidget(self.cutout_label, 11, 3)

        layout.addLayout(aug_grid)

        # Preset buttons
        preset_layout = QHBoxLayout()
        self.light_preset_btn = QPushButton("Light Augmentation")
        self.medium_preset_btn = QPushButton("Medium Augmentation")
        self.heavy_preset_btn = QPushButton("Heavy Augmentation")
        self.clear_all_btn = QPushButton("Clear All")

        self.light_preset_btn.clicked.connect(self.apply_light_preset)
        self.medium_preset_btn.clicked.connect(self.apply_medium_preset)
        self.heavy_preset_btn.clicked.connect(self.apply_heavy_preset)
        self.clear_all_btn.clicked.connect(self.clear_all_augmentations)

        preset_layout.addWidget(self.light_preset_btn)
        preset_layout.addWidget(self.medium_preset_btn)
        preset_layout.addWidget(self.heavy_preset_btn)
        preset_layout.addWidget(self.clear_all_btn)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        group.setLayout(layout)
        return group

    def create_output_section(self):
        """Create output settings section"""
        group = QGroupBox("Output Settings")
        layout = QVBoxLayout()

        # Output directory selection
        dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel("Select output directory...")
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)

        dir_layout.addWidget(QLabel("Output Directory:"))
        dir_layout.addWidget(self.output_dir_label)
        dir_layout.addWidget(self.browse_output_btn)
        layout.addLayout(dir_layout)

        # Options
        options_layout = QHBoxLayout()
        self.include_original_checkbox = QCheckBox("Include original (preprocessed)")
        self.include_original_checkbox.setChecked(True)
        options_layout.addWidget(self.include_original_checkbox)
        options_layout.addStretch()
        layout.addLayout(options_layout)

        group.setLayout(layout)
        return group

    def create_control_section(self):
        """Create control buttons and progress section"""
        group = QGroupBox("Processing")
        layout = QVBoxLayout()

        # Control buttons
        button_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Sample")
        self.process_btn = QPushButton("Process All Images")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.preview_btn.clicked.connect(self.preview_augmentations)
        self.process_btn.clicked.connect(self.process_all_images)
        self.stop_btn.clicked.connect(self.stop_processing)

        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(100)
        self.log_area.setReadOnly(True)
        layout.addWidget(QLabel("Processing Log:"))
        layout.addWidget(self.log_area)

        group.setLayout(layout)
        return group

    def apply_light_preset(self):
        """Apply light augmentation preset"""
        self.clear_all_augmentations()
        self.hflip_checkbox.setChecked(True)
        self.brightness_checkbox.setChecked(True)
        self.brightness_limit.setValue(10)
        self.contrast_checkbox.setChecked(True)
        self.contrast_limit.setValue(10)
        self.aug_count_spin.setValue(2)

    def apply_medium_preset(self):
        """Apply medium augmentation preset"""
        self.clear_all_augmentations()
        self.hflip_checkbox.setChecked(True)
        self.rotation_checkbox.setChecked(True)
        self.rotation_angle.setValue(10)
        self.brightness_checkbox.setChecked(True)
        self.brightness_limit.setValue(15)
        self.contrast_checkbox.setChecked(True)
        self.contrast_limit.setValue(15)
        self.saturation_checkbox.setChecked(True)
        self.blur_checkbox.setChecked(True)
        self.aug_count_spin.setValue(3)

    def apply_heavy_preset(self):
        """Apply heavy augmentation preset"""
        self.clear_all_augmentations()
        self.hflip_checkbox.setChecked(True)
        self.vflip_checkbox.setChecked(True)
        self.rotation_checkbox.setChecked(True)
        self.brightness_checkbox.setChecked(True)
        self.contrast_checkbox.setChecked(True)
        self.saturation_checkbox.setChecked(True)
        self.hue_checkbox.setChecked(True)
        self.blur_checkbox.setChecked(True)
        self.noise_checkbox.setChecked(True)
        self.clahe_checkbox.setChecked(True)
        self.crop_checkbox.setChecked(True)
        self.cutout_checkbox.setChecked(True)
        self.aug_count_spin.setValue(5)

    def clear_all_augmentations(self):
        """Clear all augmentation checkboxes"""
        checkboxes = [
            self.rotation_checkbox, self.hflip_checkbox, self.vflip_checkbox,
            self.brightness_checkbox, self.contrast_checkbox, self.saturation_checkbox,
            self.hue_checkbox, self.blur_checkbox, self.noise_checkbox,
            self.clahe_checkbox, self.crop_checkbox, self.cutout_checkbox
        ]
        for checkbox in checkboxes:
            checkbox.setChecked(False)

    def browse_output_directory(self):
        """Browse for output directory"""
        from PyQt5.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_label.setText(directory)

    def get_preprocessing_pipeline(self):
        """Build preprocessing pipeline based on settings"""
        transforms = []

        if self.resize_checkbox.isChecked():
            if self.maintain_aspect.isChecked():
                transforms.append(A.LongestMaxSize(max_size=max(self.resize_width.value(), self.resize_height.value())))
                transforms.append(A.PadIfNeeded(min_height=self.resize_height.value(),
                                                min_width=self.resize_width.value(),
                                                border_mode=cv2.BORDER_CONSTANT, value=0,
                                                mask_value=0))
            else:
                transforms.append(A.Resize(height=self.resize_height.value(),
                                           width=self.resize_width.value()))

        if self.grayscale_checkbox.isChecked():
            transforms.append(A.ToGray(p=1.0))

        if self.normalize_checkbox.isChecked():
            transforms.append(A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0))
        elif self.standardize_checkbox.isChecked():
            transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return A.Compose(transforms, bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    def get_augmentation_pipeline(self):
        """Build augmentation pipeline based on settings"""
        transforms = []

        # Add preprocessing first
        preprocess_transforms = self.get_preprocessing_pipeline().transforms
        transforms.extend(preprocess_transforms)

        # Add augmentations
        if self.rotation_checkbox.isChecked():
            transforms.append(A.Rotate(limit=self.rotation_angle.value(), p=0.5))

        if self.hflip_checkbox.isChecked():
            transforms.append(A.HorizontalFlip(p=self.hflip_prob.value() / 100))

        if self.vflip_checkbox.isChecked():
            transforms.append(A.VerticalFlip(p=self.vflip_prob.value() / 100))

        if self.brightness_checkbox.isChecked():
            limit = self.brightness_limit.value() / 100
            transforms.append(A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=0, p=0.5))

        if self.contrast_checkbox.isChecked():
            limit = self.contrast_limit.value() / 100
            transforms.append(A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=limit, p=0.5))

        if self.saturation_checkbox.isChecked():
            limit = self.saturation_limit.value() / 100
            transforms.append(A.HueSaturationValue(sat_shift_limit=limit, hue_shift_limit=0, val_shift_limit=0, p=0.5))

        if self.hue_checkbox.isChecked():
            limit = self.hue_limit.value() / 100
            transforms.append(A.HueSaturationValue(hue_shift_limit=limit, sat_shift_limit=0, val_shift_limit=0, p=0.5))

        if self.blur_checkbox.isChecked():
            transforms.append(A.GaussianBlur(blur_limit=(3, self.blur_limit.value()), p=0.3))

        if self.noise_checkbox.isChecked():
            transforms.append(A.GaussNoise(var_limit=(10, self.noise_limit.value()), p=0.3))

        if self.clahe_checkbox.isChecked():
            transforms.append(A.CLAHE(clip_limit=self.clahe_limit.value(), p=0.3))

        if self.crop_checkbox.isChecked():
            min_scale = self.crop_scale_min.value() / 100
            transforms.append(A.RandomResizedCrop(height=self.resize_height.value(),
                                                  width=self.resize_width.value(),
                                                  scale=(min_scale, 1.0), p=0.5))

        if self.cutout_checkbox.isChecked():
            transforms.append(A.CoarseDropout(max_holes=self.cutout_holes.value(),
                                              max_height=0.1, max_width=0.1, p=0.3))

        return A.Compose(transforms, bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    def preview_augmentations(self):
        """Preview augmentations on a sample image"""
        if not self.parent_annotator.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded to preview.")
            return

        if not hasattr(self, 'output_dir_label') or self.output_dir_label.text() == "Select output directory...":
            QMessageBox.warning(self, "Warning", "Please select an output directory first.")
            return

        # Get a sample image with annotations
        sample_index = self.parent_annotator.current_index
        sample_path = self.parent_annotator.image_paths[sample_index]

        # Create preview window
        self.create_preview_window(sample_path, sample_index)

    def create_preview_window(self, image_path, image_index):
        """Create a preview window showing original and augmented versions"""
        from PyQt5.QtWidgets import QDialog, QGridLayout, QScrollArea
        from PyQt5.QtGui import QPixmap, QImage

        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Augmentation Preview")
        preview_dialog.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()

        # Create scroll area for preview images
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        grid_layout = QGridLayout()

        # Get image annotations
        img_data = self.get_image_data(image_path, image_index)

        try:
            # Show original image
            original_label = QLabel("Original")
            original_img_label = QLabel()
            original_pixmap = QPixmap(image_path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            original_img_label.setPixmap(original_pixmap)

            grid_layout.addWidget(original_label, 0, 0, Qt.AlignCenter)
            grid_layout.addWidget(original_img_label, 1, 0)

            # Show preprocessed version
            if any([self.resize_checkbox.isChecked(), self.normalize_checkbox.isChecked(),
                    self.standardize_checkbox.isChecked(), self.grayscale_checkbox.isChecked()]):
                preprocess_pipeline = self.get_preprocessing_pipeline()
                preprocessed_img = self.apply_pipeline_to_image(img_data, preprocess_pipeline)

                if preprocessed_img is not None:
                    preprocess_label = QLabel("Preprocessed")
                    preprocess_img_label = QLabel()

                    # Convert numpy array to QPixmap
                    if len(preprocessed_img.shape) == 3:
                        h, w, ch = preprocessed_img.shape
                        bytes_per_line = ch * w
                        q_image = QImage(preprocessed_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    else:
                        h, w = preprocessed_img.shape
                        q_image = QImage(preprocessed_img.data, w, h, QImage.Format_Grayscale8)

                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    preprocess_img_label.setPixmap(scaled_pixmap)

                    grid_layout.addWidget(preprocess_label, 0, 1, Qt.AlignCenter)
                    grid_layout.addWidget(preprocess_img_label, 1, 1)

            # Show augmented versions
            augmentation_pipeline = self.get_augmentation_pipeline()

            for i in range(min(4, self.aug_count_spin.value())):  # Show up to 4 previews
                augmented_img = self.apply_pipeline_to_image(img_data, augmentation_pipeline)

                if augmented_img is not None:
                    aug_label = QLabel(f"Augmented {i + 1}")
                    aug_img_label = QLabel()

                    # Convert numpy array to QPixmap
                    if len(augmented_img.shape) == 3:
                        h, w, ch = augmented_img.shape
                        bytes_per_line = ch * w
                        q_image = QImage(augmented_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    else:
                        h, w = augmented_img.shape
                        q_image = QImage(augmented_img.data, w, h, QImage.Format_Grayscale8)

                    pixmap = QPixmap.fromImage(q_image)
                    scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    aug_img_label.setPixmap(scaled_pixmap)

                    col = 2 + (i % 3)
                    row = i // 3
                    grid_layout.addWidget(aug_label, row * 2, col, Qt.AlignCenter)
                    grid_layout.addWidget(aug_img_label, row * 2 + 1, col)

        except Exception as e:
            error_label = QLabel(f"Error generating preview: {str(e)}")
            grid_layout.addWidget(error_label, 0, 0)
            self.log_message(f"Preview error: {str(e)}")

        scroll_widget.setLayout(grid_layout)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview_dialog.close)
        layout.addWidget(close_btn)

        preview_dialog.setLayout(layout)
        preview_dialog.exec_()

    def apply_pipeline_to_image(self, img_data, pipeline):
        """Apply transformation pipeline to a single image"""
        try:
            image = cv2.imread(img_data['path'])
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prepare bounding boxes
            bboxes = []
            class_labels = []

            for bbox_data in img_data['annotations']:
                x_center, y_center, width, height = bbox_data['bbox']
                x_min = max(0, x_center - width / 2)
                y_min = max(0, y_center - height / 2)
                x_max = min(1, x_center + width / 2)
                y_max = min(1, y_center + height / 2)

                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(bbox_data['class_id'])

            # Apply transformations
            if bboxes:
                transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = pipeline(image=image)

            return transformed['image']

        except Exception as e:
            print(f"Error applying pipeline: {e}")
            return None

    def get_image_data(self, image_path, image_index):
        """Get image data with annotations"""
        # Try to load annotation data
        annotations = []

        # Check if there's a saved annotation file
        annotation_file = self.parent_annotator.get_annotation_filename(image_index)
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)

                for i, detection in enumerate(annotation_data['detections']):
                    if i in annotation_data['selected_boxes']:
                        x1, y1, x2, y2 = detection['box']
                        h, w = annotation_data['original_shape']

                        # Convert to normalized YOLO format
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h

                        # Get class ID
                        class_id = detection['cls']
                        if i in annotation_data.get('box_labels', {}):
                            custom_label = annotation_data['box_labels'][str(i)]
                            # Find class ID for custom manage_label_btn
                            for idx, name in self.parent_annotator.class_names.items():
                                if name == custom_label:
                                    class_id = idx
                                    break

                        annotations.append({
                            'bbox': [x_center, y_center, width, height],
                            'class_id': class_id
                        })
            except Exception as e:
                print(f"Error loading annotations: {e}")

        return {
            'path': image_path,
            'annotations': annotations
        }

    def process_all_images(self):
        """Process all loaded images with selected augmentations"""
        if not self.parent_annotator.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded to process.")
            return

        if not hasattr(self, 'output_dir_label') or self.output_dir_label.text() == "Select output directory...":
            QMessageBox.warning(self, "Warning", "Please select an output directory first.")
            return

        # Validate that at least some processing is selected
        has_preprocessing = any([
            self.resize_checkbox.isChecked(),
            self.normalize_checkbox.isChecked(),
            self.standardize_checkbox.isChecked(),
            self.grayscale_checkbox.isChecked()
        ])

        has_augmentation = any([
            self.rotation_checkbox.isChecked(),
            self.hflip_checkbox.isChecked(),
            self.vflip_checkbox.isChecked(),
            self.brightness_checkbox.isChecked(),
            self.contrast_checkbox.isChecked(),
            self.saturation_checkbox.isChecked(),
            self.hue_checkbox.isChecked(),
            self.blur_checkbox.isChecked(),
            self.noise_checkbox.isChecked(),
            self.clahe_checkbox.isChecked(),
            self.crop_checkbox.isChecked(),
            self.cutout_checkbox.isChecked()
        ])

        if not has_preprocessing and not has_augmentation:
            QMessageBox.warning(self, "Warning", "Please select at least one preprocessing or augmentation option.")
            return

        # Prepare output directory
        output_dir = self.output_dir_label.text()
        images_dir = os.path.join(output_dir, '../images')
        labels_dir = os.path.join(output_dir, '../labels')

        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create output directories: {e}")
            return

        # Collect all image data
        images_data = []
        annotated_count = 0

        for i, img_path in enumerate(self.parent_annotator.image_paths):
            img_data = self.get_image_data(img_path, i)
            images_data.append(img_data)
            if img_data['annotations']:
                annotated_count += 1

        if annotated_count == 0:
            reply = QMessageBox.question(self, "No Annotations",
                                         "No annotated images found. Process images without annotations?",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # Confirm processing
        aug_count = self.aug_count_spin.value() if has_augmentation else 0
        include_original = self.include_original_checkbox.isChecked()
        total_output = len(images_data) * (aug_count + (1 if include_original else 0))

        reply = QMessageBox.question(self, "Confirm Processing",
                                     f"This will process {len(images_data)} images and generate {total_output} output images.\n"
                                     f"Annotated images: {annotated_count}\n"
                                     f"Continue?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)

        if reply != QMessageBox.Yes:
            return

        # Start processing
        self.start_processing(images_data, output_dir, aug_count)

    def start_processing(self, images_data, output_dir, aug_count):
        """Start the processing worker thread"""
        # Disable process button and enable stop button
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.preview_btn.setEnabled(False)

        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing...")
        self.log_area.clear()

        # Create pipelines
        preprocess_pipeline = self.get_preprocessing_pipeline()
        augmentation_pipeline = self.get_augmentation_pipeline() if aug_count > 0 else None

        # Start worker thread
        self.worker = AugmentationWorker(
            images_data=images_data,
            augmentation_pipeline=augmentation_pipeline,
            preprocess_pipeline=preprocess_pipeline,
            output_dir=output_dir,
            augment_count=aug_count
        )

        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)

        self.worker.start()
        self.log_message(f"Started processing {len(images_data)} images...")

    def stop_processing(self):
        """Stop the processing worker thread"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        self.on_processing_finished(0, 0, stopped=True)

    def on_processing_finished(self, success_count, total_count, stopped=False):
        """Handle processing completion"""
        # Re-enable buttons
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)

        if stopped:
            self.progress_label.setText("Processing stopped by user")
            self.log_message("Processing stopped by user")
        else:
            self.progress_bar.setValue(100)
            self.progress_label.setText(f"Completed: {success_count}/{total_count} successful")
            self.log_message(f"Processing completed: {success_count}/{total_count} images processed successfully")

            if success_count > 0:
                # Show completion message
                QMessageBox.information(self, "Processing Complete",
                                        f"Successfully processed {success_count} out of {total_count} images.\n"
                                        f"Output saved to: {self.output_dir_label.text()}")

    def on_processing_error(self, error_message):
        """Handle processing error"""
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)

        self.progress_label.setText("Error occurred during processing")
        self.log_message(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{error_message}")

    def log_message(self, message):
        """Add message to log area"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")

        # Auto-scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())