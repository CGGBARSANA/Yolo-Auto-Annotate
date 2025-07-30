import os
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal


class SettingsDialog(QWidget):
    """Dialog for configuring application settings"""
    settings_saved = pyqtSignal(dict)

    def __init__(self, current_settings=None):
        super().__init__()
        self.current_settings = current_settings or {}
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        self.setWindowTitle("YOLOv11 Annotator - Settings")
        self.setGeometry(200, 200, 600, 400)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout()

        # Model path section
        model_group = QGroupBox("YOLO Model")
        model_layout = QVBoxLayout()

        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        self.browse_model_btn = QPushButton("Browse Model (.pt file)")
        self.browse_model_btn.clicked.connect(self.browse_model)

        model_layout.addWidget(QLabel("Select YOLO model file:"))
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.browse_model_btn)
        model_group.setLayout(model_layout)

        # Label directory section
        label_group = QGroupBox("Labels Directory")
        label_layout = QVBoxLayout()

        self.label_dir_label = QLabel("No directory selected")
        self.label_dir_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        self.browse_label_dir_btn = QPushButton("Browse Label Directory")
        self.browse_label_dir_btn.clicked.connect(self.browse_label_dir)

        label_layout.addWidget(QLabel("Select directory to save YOLO format labels:"))
        label_layout.addWidget(self.label_dir_label)
        label_layout.addWidget(self.browse_label_dir_btn)
        label_group.setLayout(label_layout)

        # Annotation save directory section
        annotation_group = QGroupBox("Annotations Directory")
        annotation_layout = QVBoxLayout()

        self.annotation_dir_label = QLabel("No directory selected")
        self.annotation_dir_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        self.browse_annotation_dir_btn = QPushButton("Browse Annotation Directory")
        self.browse_annotation_dir_btn.clicked.connect(self.browse_annotation_dir)

        annotation_layout.addWidget(QLabel("Select directory to save annotation files:"))
        annotation_layout.addWidget(self.annotation_dir_label)
        annotation_layout.addWidget(self.browse_annotation_dir_btn)
        annotation_group.setLayout(annotation_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.cancel_btn = QPushButton("Cancel")
        self.reset_btn = QPushButton("Reset to Defaults")

        self.save_btn.clicked.connect(self.save_settings)
        self.cancel_btn.clicked.connect(self.close)
        self.reset_btn.clicked.connect(self.reset_settings)

        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)

        # Add all to main layout
        layout.addWidget(model_group)
        layout.addWidget(label_group)
        layout.addWidget(annotation_group)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_current_settings(self):
        """Load current settings into the dialog"""
        if 'model_path' in self.current_settings:
            self.model_path_label.setText(self.current_settings['model_path'])
        if 'label_dir' in self.current_settings:
            self.label_dir_label.setText(self.current_settings['label_dir'])
        if 'annotation_save_dir' in self.current_settings:
            self.annotation_dir_label.setText(self.current_settings['annotation_save_dir'])

    def browse_model(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path_label.setText(file_path)

    def browse_label_dir(self):
        """Browse for label directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Labels Directory")
        if dir_path:
            self.label_dir_label.setText(dir_path)

    def browse_annotation_dir(self):
        """Browse for annotation directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Annotations Directory")
        if dir_path:
            self.annotation_dir_label.setText(dir_path)

    def reset_settings(self):
        """Reset all settings to empty"""
        self.model_path_label.setText("No model selected")
        self.label_dir_label.setText("No directory selected")
        self.annotation_dir_label.setText("No directory selected")

    def save_settings(self):
        """Save current settings"""
        model_path = self.model_path_label.text()
        label_dir = self.label_dir_label.text()
        annotation_dir = self.annotation_dir_label.text()

        # Validate inputs
        if model_path == "No model selected" or not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", "Please select a valid model file.")
            return

        if label_dir == "No directory selected":
            QMessageBox.warning(self, "Warning", "Please select a labels directory.")
            return

        if annotation_dir == "No directory selected":
            QMessageBox.warning(self, "Warning", "Please select an annotations directory.")
            return

        # Create directories if they don't exist
        try:
            os.makedirs(label_dir, exist_ok=True)
            os.makedirs(annotation_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creating directories: {e}")
            return

        settings = {
            'model_path': model_path,
            'label_dir': label_dir,
            'annotation_save_dir': annotation_dir
        }

        self.settings_saved.emit(settings)
        self.close()