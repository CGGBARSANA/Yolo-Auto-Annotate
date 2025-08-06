from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QMessageBox, QGroupBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import pyqtSignal
import os


class SettingsDialog(QDialog):
    settings_saved = pyqtSignal(dict)

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.setWindowTitle("Settings Configuration")
        self.setModal(True)
        self.setFixedSize(500, 400)
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Model Path Section
        model_group = QGroupBox("YOLO Model Configuration")
        model_layout = QVBoxLayout()

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select YOLO model file (.pt)")
        model_browse_btn = QPushButton("Browse Model File")
        model_browse_btn.clicked.connect(self.browse_model_path)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Path:"))
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_browse_btn)

        model_layout.addLayout(model_path_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Directory Configuration Section
        dir_group = QGroupBox("Directory Configuration")
        dir_layout = QVBoxLayout()

        # Label Directory
        self.label_dir_edit = QLineEdit()
        self.label_dir_edit.setPlaceholderText("Directory for YOLO format labels")
        label_browse_btn = QPushButton("Browse")
        label_browse_btn.clicked.connect(self.browse_label_dir)

        label_dir_layout = QHBoxLayout()
        label_dir_layout.addWidget(QLabel("Labels Directory:"))
        label_dir_layout.addWidget(self.label_dir_edit)
        label_dir_layout.addWidget(label_browse_btn)

        # Annotation Save Directory
        self.annotation_dir_edit = QLineEdit()
        self.annotation_dir_edit.setPlaceholderText("Directory for annotation JSON files")
        annotation_browse_btn = QPushButton("Browse")
        annotation_browse_btn.clicked.connect(self.browse_annotation_dir)

        annotation_dir_layout = QHBoxLayout()
        annotation_dir_layout.addWidget(QLabel("Annotations Directory:"))
        annotation_dir_layout.addWidget(self.annotation_dir_edit)
        annotation_dir_layout.addWidget(annotation_browse_btn)

        dir_layout.addLayout(label_dir_layout)
        dir_layout.addLayout(annotation_dir_layout)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Class Selection Mode Section
        class_mode_group = QGroupBox("Class Selection Mode")
        class_mode_layout = QVBoxLayout()

        self.class_mode_group = QButtonGroup()
        self.model_only_radio = QRadioButton("Model Classes Only")
        self.model_only_radio.setToolTip("Use only the classes defined in the YOLO model")

        self.custom_only_radio = QRadioButton("Custom Labels Only")
        self.custom_only_radio.setToolTip("Use only custom labels (create your own classes)")

        self.both_radio = QRadioButton("Both Model & Custom Classes")
        self.both_radio.setToolTip("Use both model classes and custom labels")
        self.both_radio.setChecked(True)  # Default selection

        self.class_mode_group.addButton(self.model_only_radio, 0)
        self.class_mode_group.addButton(self.custom_only_radio, 1)
        self.class_mode_group.addButton(self.both_radio, 2)

        class_mode_layout.addWidget(self.model_only_radio)
        class_mode_layout.addWidget(self.custom_only_radio)
        class_mode_layout.addWidget(self.both_radio)

        # Add description label
        description_label = QLabel(
            "• Model Classes Only: Restricts labeling to predefined model classes\n"
            "• Custom Labels Only: Allows only user-defined custom labels\n"
            "• Both: Allows using both model classes and custom labels (recommended)"
        )
        description_label.setStyleSheet("color: gray; font-size: 9pt; margin-top: 10px;")
        class_mode_layout.addWidget(description_label)

        class_mode_group.setLayout(class_mode_layout)
        layout.addWidget(class_mode_group)

        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        cancel_btn = QPushButton("Cancel")

        save_btn.clicked.connect(self.save_settings)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_current_settings(self):
        """Load current settings into the form"""
        if not self.current_settings:
            return

        # Load paths
        self.model_path_edit.setText(self.current_settings.get('model_path', ''))
        self.label_dir_edit.setText(self.current_settings.get('label_dir', ''))
        self.annotation_dir_edit.setText(self.current_settings.get('annotation_save_dir', ''))

        # Load class selection mode
        mode = self.current_settings.get('class_selection_mode', 'both')
        if mode == 'model_only':
            self.model_only_radio.setChecked(True)
        elif mode == 'custom_only':
            self.custom_only_radio.setChecked(True)
        else:
            self.both_radio.setChecked(True)

    def browse_model_path(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model File",
            "",
            "PyTorch Models (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_label_dir(self):
        """Browse for labels directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Labels Directory"
        )
        if dir_path:
            self.label_dir_edit.setText(dir_path)

    def browse_annotation_dir(self):
        """Browse for annotation save directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Annotations Directory"
        )
        if dir_path:
            self.annotation_dir_edit.setText(dir_path)

    def get_selected_class_mode(self):
        """Get the selected class selection mode"""
        if self.model_only_radio.isChecked():
            return 'model_only'
        elif self.custom_only_radio.isChecked():
            return 'custom_only'
        else:
            return 'both'

    def validate_settings(self):
        """Validate the entered settings"""
        model_path = self.model_path_edit.text().strip()
        label_dir = self.label_dir_edit.text().strip()
        annotation_dir = self.annotation_dir_edit.text().strip()

        # Check if all required fields are filled
        if not model_path:
            QMessageBox.warning(self, "Validation Error", "Please select a YOLO model file.")
            return False

        if not label_dir:
            QMessageBox.warning(self, "Validation Error", "Please select a labels directory.")
            return False

        if not annotation_dir:
            QMessageBox.warning(self, "Validation Error", "Please select an annotations directory.")
            return False

        # Check if model file exists
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Validation Error", "The selected model file does not exist.")
            return False

        # Check if model file has correct extension
        if not model_path.lower().endswith('.pt'):
            QMessageBox.warning(self, "Validation Error", "Please select a valid PyTorch model file (.pt).")
            return False

        return True

    def save_settings(self):
        """Save the settings"""
        if not self.validate_settings():
            return

        settings = {
            'model_path': self.model_path_edit.text().strip(),
            'label_dir': self.label_dir_edit.text().strip(),
            'annotation_save_dir': self.annotation_dir_edit.text().strip(),
            'class_selection_mode': self.get_selected_class_mode()
        }

        # Create directories if they don't exist
        try:
            os.makedirs(settings['label_dir'], exist_ok=True)
            os.makedirs(settings['annotation_save_dir'], exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create directories: {e}")
            return

        self.settings_saved.emit(settings)
        self.accept()