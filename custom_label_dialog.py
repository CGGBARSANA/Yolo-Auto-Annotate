from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
    QInputDialog, QMessageBox, QDialogButtonBox, QLabel
)
from PyQt5.QtCore import Qt


class CustomLabelDialog(QDialog):
    def __init__(self, label_manager, parent=None):
        super().__init__(parent)
        self.label_manager = label_manager
        self.setWindowTitle("Manage Custom Labels")
        self.setModal(True)
        self.resize(400, 300)
        self.setup_ui()
        self.refresh_list()

    def setup_ui(self):
        # Create new layout instance to avoid parent issues
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Custom Labels Management")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # List widget
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # Buttons layout
        button_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Label")
        self.edit_btn = QPushButton("Edit Label")
        self.remove_btn = QPushButton("Remove Label")
        self.refresh_btn = QPushButton("Refresh")

        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.edit_btn)
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.refresh_btn)

        layout.addLayout(button_layout)

        # Dialog buttons
        dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(dialog_buttons)

        # Set the layout
        self.setLayout(layout)

        # Connect signals
        self.add_btn.clicked.connect(self.add_label)
        self.edit_btn.clicked.connect(self.edit_label)
        self.remove_btn.clicked.connect(self.remove_label)
        self.refresh_btn.clicked.connect(self.refresh_list)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)

    def refresh_list(self):
        """Refresh the list of custom labels"""
        self.list_widget.clear()
        custom_labels = self.label_manager.get_custom_labels_list()

        if not custom_labels:
            self.list_widget.addItem("No custom labels defined")
        else:
            # Get the appropriate labels dictionary based on current mode
            if self.label_manager.current_mode == "custom_only":
                labels_dict = self.label_manager.custom_only_labels
            else:
                labels_dict = self.label_manager.custom_labels

            for label in custom_labels:
                class_id = labels_dict.get(label, 0)
                self.list_widget.addItem(f"{label} (ID: {class_id})")

    def add_label(self):
        """Add a new custom label"""
        text, ok = QInputDialog.getText(self, 'Add Custom Label', 'Enter label name:')
        if ok and text.strip():
            label_name = text.strip()
            if label_name in self.label_manager.custom_labels:
                QMessageBox.warning(self, "Duplicate Label",
                                    f"Label '{label_name}' already exists.")
                return

            try:
                class_id = self.label_manager.add_custom_label(label_name)
                QMessageBox.information(self, "Success",
                                        f"Added label '{label_name}' with ID {class_id}")
                self.refresh_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error adding label: {e}")

    def edit_label(self):
        """Edit selected custom label"""
        current_item = self.list_widget.currentItem()
        if not current_item or current_item.text() == "No custom labels defined":
            QMessageBox.warning(self, "No Selection", "Please select a label to edit.")
            return

        # Extract label name from the list item text
        item_text = current_item.text()
        if " (ID: " in item_text:
            old_name = item_text.split(" (ID: ")[0]
        else:
            old_name = item_text

        text, ok = QInputDialog.getText(self, 'Edit Custom Label',
                                        'Enter new label name:', text=old_name)
        if ok and text.strip():
            new_name = text.strip()
            if new_name == old_name:
                return

            if new_name in self.label_manager.custom_labels:
                QMessageBox.warning(self, "Duplicate Label",
                                    f"Label '{new_name}' already exists.")
                return

            try:
                if self.label_manager.edit_custom_label(old_name, new_name):
                    QMessageBox.information(self, "Success",
                                            f"Renamed '{old_name}' to '{new_name}'")
                    self.refresh_list()
                else:
                    QMessageBox.warning(self, "Error", "Failed to edit label.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error editing label: {e}")

    def remove_label(self):
        """Remove selected custom label"""
        current_item = self.list_widget.currentItem()
        if not current_item or current_item.text() == "No custom labels defined":
            QMessageBox.warning(self, "No Selection", "Please select a label to remove.")
            return

        # Extract label name from the list item text
        item_text = current_item.text()
        if " (ID: " in item_text:
            label_name = item_text.split(" (ID: ")[0]
        else:
            label_name = item_text

        reply = QMessageBox.question(self, 'Remove Label',
                                     f'Are you sure you want to remove "{label_name}"?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                if self.label_manager.remove_custom_label(label_name):
                    QMessageBox.information(self, "Success",
                                            f"Removed label '{label_name}'")
                    self.refresh_list()
                else:
                    QMessageBox.warning(self, "Error", "Failed to remove label.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing label: {e}")