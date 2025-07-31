# image_thumbnail.py - Enhanced version with selection support

import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, QRect


class ImageThumbnail(QWidget):
    clicked = pyqtSignal(int)
    selection_changed = pyqtSignal()  # New signal for selection changes

    def __init__(self, image_path, index):
        super().__init__()
        self.image_path = image_path
        self.index = index
        self.is_current = False
        self.is_annotated = False
        self.is_selected = False  # New selection state

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)

        # Checkbox for selection
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.on_selection_changed)
        layout.addWidget(self.checkbox)

        # Image label
        self.image_label = QLabel()
        self.image_label.setFixedSize(150, 150)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")

        # Load and set thumbnail image
        self.load_thumbnail()
        layout.addWidget(self.image_label)

        # Filename label
        self.filename_label = QLabel(os.path.basename(self.image_path))
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setWordWrap(True)
        self.filename_label.setMaximumHeight(40)
        layout.addWidget(self.filename_label)

        self.setLayout(layout)
        self.setFixedSize(170, 220)  # Increased height to accommodate checkbox

    def load_thumbnail(self):
        """Load and display thumbnail image"""
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Scale image to fit label while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Invalid\nImage")
        except Exception as e:
            self.image_label.setText(f"Error\nLoading\n{str(e)[:20]}")

    def on_selection_changed(self, state):
        """Handle checkbox state change"""
        self.is_selected = (state == Qt.Checked)
        self.update_border()
        self.selection_changed.emit()

    def set_selected(self, selected):
        """Set selection state programmatically"""
        self.is_selected = selected
        self.checkbox.setChecked(selected)
        self.update_border()

    def set_current(self, current):
        """Set as current image"""
        self.is_current = current
        self.update_border()

    def set_annotated(self, annotated):
        """Set annotation status"""
        self.is_annotated = annotated
        self.update_border()

    def update_border(self):
        """Update border style based on current state"""
        if self.is_current:
            border_color = "blue"
            border_width = 4
        elif self.is_selected:
            border_color = "orange"
            border_width = 3
        elif self.is_annotated:
            border_color = "green"
            border_width = 2
        else:
            border_color = "gray"
            border_width = 2

        self.image_label.setStyleSheet(f"border: {border_width}px solid {border_color};")

        # Update filename label color for better visibility
        if self.is_current:
            self.filename_label.setStyleSheet("color: blue; font-weight: bold;")
        elif self.is_selected:
            self.filename_label.setStyleSheet("color: orange; font-weight: bold;")
        elif self.is_annotated:
            self.filename_label.setStyleSheet("color: green;")
        else:
            self.filename_label.setStyleSheet("color: black;")

    def mousePressEvent(self, event):
        """Handle mouse click on thumbnail"""
        if event.button() == Qt.LeftButton:
            # Don't trigger click if clicking on checkbox
            if not self.checkbox.geometry().contains(event.pos()):
                self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to toggle selection"""
        if event.button() == Qt.LeftButton:
            self.set_selected(not self.is_selected)
        super().mouseDoubleClickEvent(event)